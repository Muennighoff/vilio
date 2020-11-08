import collections
import os
import random

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from param import args
from fts_lmdb.hm_pretrain_data import InputExample, LXMERTDataset, LXMERTTorchDataset 
from fts_lmdb.hm_data import HMEvaluator
from utils.pandas_scripts import clean_data
from src.vilio.transformers.tokenization_auto import AutoTokenizer
from src.vilio.transformers.optimization import AdamW, get_linear_schedule_with_warmup
from src.vilio.modeling_robertaV import RobertaVPretraining


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
    
    evaluator = HMEvaluator(tset)
    print()

    return DataTuple(dataset=dset, torchdset=tset, loader=data_loader, evaluator=evaluator)

# Create pretrain.jsonl & traindev data
clean_data("./data")

train_tuple = get_tuple(args.train, args.batch_size, shuffle=True, drop_last=True)

if args.valid != "":
    valid_bsize = 500 
    valid_tuple = get_tuple(
        args.valid, bs=valid_bsize,
        shuffle=False, drop_last=False
    )
else:
    valid_tuple = None


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids, input_mask, segment_ids, lm_label_ids,
                 visual_feats, obj_labels,
                 is_matched, ans, vl_label):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids

        self.visual_feats = visual_feats
        self.obj_labels = obj_labels

        self.is_matched = is_matched

        self.ans = ans

        self.vl_label = vl_label

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
                print("Not found:", token)
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab["<unk>"]) # Not [UNK] as in Bert
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label

def mask_tokens(inputs: torch.Tensor, tokenizer):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    mlm_probability = 0.15

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -1 #00  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels

def random_feat(feats):
    mask_feats = feats.clone() #copy()
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
    
    # Account for <s> and </s> </s> with "- 3"
    if len(tokens) > max_seq_length - 3:
        tokens = tokens[:(max_seq_length - 3)]

    # We are not performing pretraining on the very first id, as the original roberta was pretrained with cased words & no space in front
    # We also use a huggingface implementation of masking here (It doesnt make any big difference in the loss landscape though)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    first_id = input_ids[0]

    if len(input_ids) < 2:
        input_ids, masked_label = [], []
    else:
        masked_tokens, masked_label = mask_tokens(torch.tensor([input_ids[1:]]), tokenizer)
        input_ids, masked_label = masked_tokens.tolist()[0], masked_label.tolist()[0]

    input_ids = [0] + [first_id] + input_ids + [2] + [2]

    # Mask & Segment Word
    lm_label_ids = ([-1] + [-1] + masked_label + [-1]) 
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)

    # As VisualBERT concats Text & Visual Input, lm label ids must be even longer!
    num_features = 100  # 100 features for Hateful Memes!
    while len(lm_label_ids) < (max_seq_length + num_features):
        lm_label_ids.append(-1)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(1)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(lm_label_ids) == max_seq_length + num_features

    feat, boxes = example.visual_feats
    obj_labels, obj_confs = example.obj_labels
    attr_labels, attr_confs = example.attr_labels

    # Mask Image Features:
    masked_feat, feat_mask = random_feat(feat)

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
        vl_label=example.vl_label
    )
    return features

LOSSES_NAME = ('Mask_LM', 'VL', 'Matched', 'Feat', 'Obj', 'QA') 

## I.e. : Mask_LM = Masking words; 
# Obj, Feat = Masking objs (ids), feats (pixels?), 
# Matched = Sen & Img belong together? 

class LXMERT:
    def __init__(self, max_seq_length):
        super().__init__()
        self.max_seq_length = max_seq_length

        self.tokenizer = AutoTokenizer.from_pretrained(args.tr, do_lower_case=True)

        # Build model
        self.model = RobertaVPretraining(
            args.tr,
            visual_losses=args.visual_losses,
            task_matched=args.task_matched,
            task_obj_predict=args.task_obj_predict,
            task_hm=args.task_hm
        )

        # Update config to finetune token type embeddings for Roberta
        if self.model.roberta.config.type_vocab_size == 1:
            print("Type Vocab Size is 1. Adjusting...")
            self.model.roberta.config.type_vocab_size = 2 
            # Create a new Embeddings layer, with 2 possible segments IDs instead of 1
            self.model.roberta.embeddings.token_type_embeddings = nn.Embedding(2, self.model.roberta.config.hidden_size)             
            # Initialize it
            self.model.roberta.embeddings.token_type_embeddings.weight.data.normal_(mean=0.0, std=self.model.roberta.config.initializer_range)

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
        segment_ids = None #torch.tensor([f.segment_ids for f in train_features], dtype=torch.long).cuda()

        # Visual Inputs
        feats = torch.from_numpy(np.stack([f.visual_feats[0] for f in train_features])).cuda()
        pos = torch.from_numpy(np.stack([f.visual_feats[1] for f in train_features])).cuda()

        # Language Prediction
        lm_labels = torch.tensor([f.lm_label_ids for f in train_features], dtype=torch.long).cuda()

        # Visual Prediction
        if args.task_obj_predict:
            obj_labels = {}
            # Removed 'attr',
            for key in ('obj', 'feat'):  
                visn_labels = torch.from_numpy(np.stack([f.obj_labels[key][0] for f in train_features])).cuda()
                visn_mask = torch.from_numpy(np.stack([f.obj_labels[key][1] for f in train_features])).cuda()
                assert visn_labels.size(0) == visn_mask.size(0) and visn_labels.size(1) == visn_mask.size(1)
                obj_labels[key] = (visn_labels, visn_mask)
        else:   
            obj_labels = None

        # Joint Prediction
        if args.task_matched:
            torch.tensor([f.is_matched for f in train_features], dtype=torch.long).cuda()
        else:
            matched_labels = None

        if args.task_qa:
            ans = torch.from_numpy(np.stack([f.ans for f in train_features])).cuda()

        # Label
        if args.task_hm:
            vl_label = torch.tensor([f.vl_label for f in train_features], dtype=torch.long).cuda()
        else:
            vl_label = None

        """
        forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                visual_feats=None, pos=None, obj_labels=None, matched_label=None, ans=None):
        """

        loss, losses, ans_logit = self.model(
            input_ids, attention_mask=input_mask, visual_embeddings=feats, position_embeddings_visual=pos, masked_lm_labels=lm_labels,
            matched_label=matched_labels, obj_labels=obj_labels, vl_label=vl_label
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
            # Somehow had to add [0] here, which is not in or repo
            for name, loss in zip(LOSSES_NAME, total_losses[0]):
                losses_str += "%s: %0.4f " % (name, loss / batch_per_epoch)
            print(losses_str)

            if eval_tuple is not None and args.task_hm:
                self.evaluate(eval_tuple)

            if args.task_qa:
                train_tuple.evaluator.evaluate(uid2ans, pprint=True)
                
            if epoch == 10:
                self.save("Epoch%02d" % (epoch+1))

        self.save("LAST")

    def predict(self, eval_tuple: DataTuple, dump=None, out_csv=True):

        self.model.eval()

        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        quesid2prob = {}

        for i, batch in enumerate(loader):

            loss, losses, logit = self.valid_batch(batch)

            # Note: LogSoftmax does not change order, hence there should be nothing wrong with taking it as our prediction
            logit = self.logsoftmax(logit)
            score = logit[:, 1]

            _, predict = logit.max(1)

            for qid, l in zip(ques_id, predict.cpu().numpy()):
                quesid2ans[qid] = l

            # Getting probas for Roc Auc
            for qid, l in zip(ques_id, score.cpu().numpy()):
                quesid2prob[qid] = l

        return quesid2ans, quesid2prob

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        quesid2ans, quesid2prob = self.predict(eval_tuple, dump=dump)

        acc = eval_tuple.evaluator.evaluate(quesid2ans)
        roc_auc = eval_tuple.evaluator.roc_auc(quesid2prob)

        print("SCORES ARE: ", acc, roc_auc)



    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(args.output, "%s_RV.pth" % name))

    def load(self, path):
        print("Load BERT extractor from %s" % path)
        state_dict = torch.load("%s" % path)
        self.model.load_state_dict(state_dict)

    def loadpre(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s" % path)

        # Do not load any answer head
        for key in list(state_dict.keys()):
            if 'answer' in key:
                state_dict.pop(key)

        # Change Multi GPU to single GPU
        new_state_dict = {}
        for key, value in state_dict.items():
            
            # Skip TTIs
            if "embeddings.token_type_embeddings.weight" in key:
                continue

            if key.startswith("model.bert."):
                print("SAVING {} as {}.".format(key, key[6:])) # Save as bert. ...
                new_state_dict[key[6:]] = value
            elif key.startswith("module."):
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





    