import collections
import os
import random

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from param import args

if args.tsv:
    from fts_tsv.pretrain_data_tsv import InputExample, LXMERTDataset, LXMERTTorchDataset
else:
    from fts_lmdb.pretrain_data import InputExample, LXMERTDataset, LXMERTTorchDataset 
from utils.pandas_scripts import clean_data
from src.vilio.transformers.tokenization_auto import AutoTokenizer
from src.vilio.transformers.optimization import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from src.vilio.modeling_robertaX import RobertaXPretraining


DataTuple = collections.namedtuple("DataTuple", 'dataset torchdset loader evaluator')

def get_tuple(splits: str, bs: int, shuffle=False, drop_last=False, topk=-1) -> DataTuple:
    # Decide which QA datasets would be used in pre-training.
    # Options: vqa, gqa, visual7w
    # Note: visual7w is a part of vgqa, we take the name here.
    qa_sets = args.qa_sets
    if qa_sets is not None:
        qa_sets = set(qa_set.lower().strip() for qa_set in qa_sets.split(","))
    
    print(splits)

    # Build dataset, data loader, and evaluator.
    dset = LXMERTDataset(splits)
    tset = LXMERTTorchDataset(splits) # Remove topk
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        collate_fn=lambda x: x,
        drop_last=drop_last, pin_memory=True
    )
    evaluator = None
    print()

    return DataTuple(dataset=dset, torchdset=tset, loader=data_loader, evaluator=evaluator)

# Create pretrain.jsonl & traindev data
clean_data("./data")

train_tuple = get_tuple(args.train, args.batch_size, shuffle=True, drop_last=True)
valid_tuple = None

#valid_batch_size = 2048 if args.multiGPU else 512
#valid_tuple = get_tuple(args.valid, valid_batch_size, shuffle=False, drop_last=False, topk=5000)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids, input_mask, segment_ids, lm_label_ids,
                 visual_feats, obj_labels,
                 is_matched, ans):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids

        self.visual_feats = visual_feats
        self.obj_labels = obj_labels

        self.is_matched = is_matched

        self.ans = ans

def random_word(tokens, tokenizer):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):

        prob = random.random()
        # mask token with probability
        ratio = args.word_mask_rate
        if prob < ratio:
            prob /= ratio

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "<mask>" # Not [MASK] as in normal Bert

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.get_vocab().items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.get_vocab()[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab["<unk>"]) # Not [UNK] as in Bert
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label


def random_feat(feats):
    if args.tsv:
        mask_feats = feats.copy()
    else:
        mask_feats = feats.clone() #copy() #clone() #copy()
    feat_mask = np.zeros(len(feats), dtype=np.float32)
    for i in range(len(feats)):
        prob = random.random()
        # mask token with probability
        if prob < args.obj_mask_rate:
            prob /= args.obj_mask_rate

            # 80% randomly change token to zero feat
            if prob < 0.8:
                mask_feats[i, :] = 0.

            # 10% randomly change token to random feat
            elif prob < 0.9:
                mask_feats[i, :] = train_tuple.torchdset.random_feat()
            # -> rest 10% randomly keep current feat

            # Need to predict this feat
            feat_mask[i] = 1.

    return mask_feats, feat_mask


def convert_example_to_features(example: InputExample, max_seq_length, tokenizer)->InputFeatures:
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, input_mask, CLS and SEP tokens etc.
    :param example: InputExample, containing sentence input as strings and is_next label
    :param max_seq_length: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """
    tokens = tokenizer.tokenize(" " + " ".join(str(example.sent).split()))

    # Account for <s> and </s> with "- 2"
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[:(max_seq_length - 2)]

    # Get random words
    masked_tokens, masked_label = random_word(tokens, tokenizer)

    input_ids = tokenizer.convert_tokens_to_ids(masked_tokens)
    input_ids = [0] + input_ids + [2]

    # Mask & Segment Word
    lm_label_ids = ([-1] + masked_label + [-1])
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(1) # Roberta padding == 1
        input_mask.append(0)
        segment_ids.append(0)
        lm_label_ids.append(-1)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(lm_label_ids) == max_seq_length

    feat, boxes = example.visual_feats
    obj_labels, obj_confs = example.obj_labels
    attr_labels, attr_confs = example.attr_labels

    # Mask Image Features:
    masked_feat, feat_mask = random_feat(feat)

    # No QA pretraining
    ans = -1

    features = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        lm_label_ids=lm_label_ids,
        visual_feats=(masked_feat, boxes),
        obj_labels={
            'obj': (obj_labels, obj_confs),
            'attr': (attr_labels, attr_confs),
            'feat': (feat, feat_mask),
        },
        is_matched=example.is_matched,
        ans=ans,
    )
    return features


LOSSES_NAME = ('Mask_LM', 'Matched', 'Obj', 'Feat') #Removed 'Attr', 'QA'

## I.e. : Mask_LM = Masking words; 
# Obj, Feat = Masking objs (ids), feats (pixels?), 
# Matched = Sen & Img belong together? 


class LXMERT:
    def __init__(self, max_seq_length):
        super().__init__()
        self.max_seq_length = max_seq_length

        self.tokenizer = AutoTokenizer.from_pretrained(args.tr, do_lower_case=True)

        # Build model
        self.model, loading_info = RobertaXPretraining.from_pretrained(
            args.tr,
            llayers=args.llayers,
            xlayers=args.xlayers, 
            rlayers=args.rlayers,
            task_mask_lm=args.task_mask_lm,
            task_obj_predict=args.task_obj_predict,
            task_matched=args.task_matched,
            task_qa=args.task_qa,
            visual_losses=args.visual_losses,
            num_answers=None,
            output_loading_info=True
        )

        print("UNEXPECTED: ", loading_info["unexpected_keys"])
        print("MISSING: ", loading_info["missing_keys"])
        print("ERRORS: ", loading_info["error_msgs"])

        # Weight initialization and loading
        if args.from_scratch:
            print("Train from Scratch: re-initialize all BERT weights.")
            self.model.apply(self.model.init_bert_weights)
        if args.loadfin is not None:
            self.load(args.loadfin)
        if args.loadpre is not None:
            self.loadpre(args.loadpre)

        # GPU Options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model = nn.DataParallel(self.model)


    def forward(self, examples):
        train_features = [convert_example_to_features(example, self.max_seq_length, self.tokenizer)
                          for example in examples]

        # language Inputs
        input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).cuda()
        input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long).cuda()
        segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long).cuda()

        # Visual Inputs
        feats = torch.from_numpy(np.stack([f.visual_feats[0] for f in train_features])).cuda()
        pos = torch.from_numpy(np.stack([f.visual_feats[1] for f in train_features])).cuda()

        # Language Prediction
        lm_labels = torch.tensor([f.lm_label_ids for f in train_features], dtype=torch.long).cuda()

        # Visual Prediction
        obj_labels = {}
        # Removed 'attr',
        for key in ('obj', 'feat'):  
            visn_labels = torch.from_numpy(np.stack([f.obj_labels[key][0] for f in train_features])).cuda()
            visn_mask = torch.from_numpy(np.stack([f.obj_labels[key][1] for f in train_features])).cuda()
            assert visn_labels.size(0) == visn_mask.size(0) and visn_labels.size(1) == visn_mask.size(1)
            obj_labels[key] = (visn_labels, visn_mask)

        # Joint Prediction
        matched_labels = torch.tensor([f.is_matched for f in train_features], dtype=torch.long).cuda()
        ans = torch.from_numpy(np.stack([f.ans for f in train_features])).cuda()

        """
        forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                visual_feats=None, pos=None, obj_labels=None, matched_label=None, ans=None):
        """

        loss, losses, ans_logit = self.model(
            input_ids, segment_ids, input_mask, lm_labels,
            feats, pos, obj_labels, matched_labels, ans
        )

        return loss, losses.detach().cpu(), ans_logit

    def train_batch(self, optim, scheduler, batch, ups):
        loss, losses, ans_logit = self.forward(batch)
        if args.multiGPU:
            loss = loss.mean()
            losses = losses.mean(0)

        # Account for grad accum.
        loss /= args.acc
        losses /= args.acc

        loss.backward()
        
        if (ups+1) % args.acc == 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
            optim.step()
            scheduler.step()
            optim.zero_grad()

        return loss.item(), losses.cpu().numpy(), ans_logit

    def valid_batch(self, batch):
        with torch.no_grad():
            loss, losses, ans_logit = self.forward(batch)
            if args.multiGPU:
                loss = loss.mean()
                losses = losses.mean(0)
        return loss.item(), losses.cpu().numpy(), ans_logit


    def train(self, train_tuple: DataTuple, eval_tuple: DataTuple):
        train_ld = train_tuple.loader

        # Optimizer
        batch_per_epoch = len(train_ld)
        t_total = int(batch_per_epoch * args.epochs // args.acc)
        warmup_ratio = 0.05
        warmup_iters = int(t_total * warmup_ratio)

        print("Batch per epoch: %d" % batch_per_epoch)
        print("Total Iters: %d" % t_total)
        print("Warm up Iters: %d" % warmup_iters)

        optim = AdamW(self.model.parameters(), lr=args.lr)
        scheduler = get_linear_schedule_with_warmup(optim, warmup_iters, t_total)
        optim.zero_grad()

        # Tracking updates for accumulation
        ups = 0

        # Train
        best_eval_loss = 9595.
        for epoch in range(args.epochs):
            # Train
            self.model.train()
            total_loss = 0.
            total_losses = 0.
            uid2ans = {}
            for batch in tqdm(train_ld, total=len(train_ld)):
                loss, losses, logit = self.train_batch(optim, scheduler, batch, ups)
                total_loss += loss
                total_losses += losses

                ups += 1

                if args.task_qa:
                    score, label = logit.max(1)
                    for datum, l in zip(batch, label.cpu().numpy()):
                        uid = datum.uid
                        ans = train_tuple.dataset.answer_table.id2ans(l)
                        uid2ans[uid] = ans

            print("The training loss for Epoch %d is %0.4f" % (epoch, total_loss / (batch_per_epoch * args.acc)))
            losses_str = "The losses are "
            # Somehow had to add [0] here, which is not in or. repo
            for name, loss in zip(LOSSES_NAME, total_losses[0]):
                losses_str += "%s: %0.4f " % (name, loss / batch_per_epoch)
            print(losses_str)

            # Save one halfway
            if epoch == 5:
                self.save("Epoch%02d" % (epoch+1))

        self.save("LAST")


    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(args.output, "%s_RX.pth" % name))

    def load(self, path):
        print("Load BERT extractor from %s" % path)
        state_dict = torch.load("%s" % path)
        self.model.load_state_dict(state_dict)

    def loadpre(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s" % path) # Removed _LXRT.pth

        # Do not load any answer head
        for key in list(state_dict.keys()):
            if 'answer' in key:
                state_dict.pop(key)

        # Change Multi GPU to single GPU
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[len("module."):]] = value
            else:
                new_state_dict[key] = value

        state_dict = new_state_dict

        load_keys = set(state_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        print()
        print("Keys in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Keys in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()

        self.model.load_state_dict(state_dict, strict=False)


if __name__ == "__main__":

    lxmert = LXMERT(max_seq_length=128)

    lxmert.train(train_tuple, valid_tuple)





    