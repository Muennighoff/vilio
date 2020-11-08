import os

import torch
import torch.nn as nn

from param import args

from src.vilio.modeling_bertX import BertLayerNorm, GeLU, BertLayer
from src.vilio.modeling_albertX import GeLU_new

from src.vilio.modeling_bertU import BertU
from src.vilio.modeling_robertaU import RobertaU

from src.vilio.transformers.tokenization_auto import AutoTokenizer

from torch.nn.utils.rnn import pad_sequence


### TOKENIZER NOTES:
###       In Roberta: [0] == <s> ;;; [1] == <pad> ;;; [2] == </s> ;;; [50264] == <mask>
###       In BERT:  [CLS]  [PAD]   [SEP]    [MASK] 
###       In ALBERT: [CLS]  <pad>  [PAD]    [MASK]
### https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-vocab.json
### https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt
### Segment_ids is the same as token_type_ids


class ModelU(nn.Module):
    def __init__(self, args=args, max_seq_len=100+args.num_features, num_features=args.num_features, tr_name=args.tr):
        """
        max_seq_len: How long our input will be in total!
        num_features: How much of that input will be bboxes
        tr_name: bert-... / roberta-...

        To not loose any input, make sure num_features + longest input sentence < max_seq_len.
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self.num_features = num_features
        self.tr_name = tr_name

        ### BUILD TOKENIZER ###
        self.tokenizer = AutoTokenizer.from_pretrained(self.tr_name)

        ### BUILD MODEL ###
        # Note: Specific for HM Dataset the dim is 2048
        if tr_name.startswith("roberta"):
            self.model, loading_info = RobertaU.from_pretrained(self.tr_name, img_dim=2048, output_loading_info=True)
        elif tr_name.startswith("bert"):
            self.model, loading_info = BertU.from_pretrained(self.tr_name, img_dim=2048, output_loading_info=True)

        print("UNEXPECTED: ", loading_info["unexpected_keys"])
        print("MISSING: ", loading_info["missing_keys"])
        print("ERRORS: ", loading_info["error_msgs"])

        # Update config to finetune token type embeddings for Roberta
        if self.model.config.type_vocab_size == 1:
            print("Type Vocab Size is 1. Adjusting...")
            self.model.config.type_vocab_size = 2 
            # Create a new Embeddings layer, with 2 possible segments IDs instead of 1
            self.model.embeddings.token_type_embeddings = nn.Embedding(2, self.model.config.hidden_size)             
            # Initialize it
            self.model.embeddings.token_type_embeddings.weight.data.normal_(mean=0.0, std=self.model.config.initializer_range)


        ### CLASSIFICATION HEADS ###
        # Note: LXRT Default classifier tends to perform best; For Albert gelu_new outperforms gelu
        # LXRTModel Default classifier (Note: This classifier is also used in UNITER!)

        self.classifier = nn.Sequential(
            nn.Linear(self.dim, self.dim * 2),
            GeLU(),
            BertLayerNorm(self.dim * 2, eps=1e-12),
            nn.Linear(self.dim * 2, 2)
        )
        self.classifier.apply(self.init_weights)

        if args.from_scratch:
            print("initializing all the weights")
            self.model.apply(self.model.init_weights)
        
    @property
    def dim(self):
        return self.model.config.hidden_size

    def pad_tensors(self, tensors, lens=None, pad=0):
        """Copied from UNITER Repo --- B x [T, ...]"""
        if lens is None:
            lens = [t.size(0) for t in tensors]
        max_len = max(lens)
        bs = len(tensors)
        hid = tensors[0].size(-1)
        dtype = tensors[0].dtype
        output = torch.zeros(bs, max_len, hid, dtype=dtype)
        if pad:
            output.data.fill_(pad)
        for i, (t, l) in enumerate(zip(tensors, lens)):
            output.data[i, :l, ...] = t.data
        return output

    def preprocess_roberta(self, sents, visual_feats, num_features, tokenizer):
        """
        Copied & adapted from UNITER Repo.
        """

        iids = []
        attn_masks = []

        for (i, sent) in enumerate(sents):

            sent = " " + " ".join(str(sent).split())
            tokens = tokenizer.tokenize(sent)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_ids = [0] + input_ids + [2] + [2] + [0]

            attn_mask = [1] * (len(input_ids) + num_features)

            input_ids = torch.tensor(input_ids)
            attn_mask = torch.tensor(attn_mask)

            iids.append(input_ids)
            attn_masks.append(attn_mask)


        txt_lens = [i.size(0) for i in iids]

        input_ids = pad_sequence(iids, batch_first=True, padding_value=1) # [1] is Roberta's padding token
        attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

        img_feats, img_pos_feats = visual_feats

        # image batches
        num_bbs = [f.size(0) for f in img_feats]
        img_feats = self.pad_tensors(img_feats, num_bbs)
        img_pos_feats = self.pad_tensors(img_pos_feats, num_bbs)

        bs, max_tl = input_ids.size()
        out_size = attn_masks.size(1)
        gather_index = self.get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

        return input_ids, img_feats, img_pos_feats, attn_masks, gather_index


    def preprocess_bert(self, sents, visual_feats, num_features, tokenizer):
        """
        Copied & adapted from UNITER Repo.
        """
        iids = []
        attn_masks = []

        for (i, sent) in enumerate(sents):

            sent = " ".join(str(sent).split())
            
            # Case first letter for bert-base-cased Uniter performs better
            if args.case:
                sent = sent[0].upper() + sent[1:]

            tokens = tokenizer.tokenize(sent)

            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            attn_mask = [1] * (len(input_ids) + num_features)

            input_ids = torch.tensor(input_ids)
            attn_mask = torch.tensor(attn_mask)

            iids.append(input_ids)
            attn_masks.append(attn_mask)


        txt_lens = [i.size(0) for i in iids]

        input_ids = pad_sequence(iids, batch_first=True, padding_value=0)
        attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)

        img_feats, img_pos_feats = visual_feats
        # image batches
        num_bbs = [f.size(0) for f in img_feats]

        img_feats = self.pad_tensors(img_feats, num_bbs)
        img_pos_feats = self.pad_tensors(img_pos_feats, num_bbs)

        bs, max_tl = input_ids.size()
        out_size = attn_masks.size(1)

        gather_index = self.get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

        return input_ids, img_feats, img_pos_feats, attn_masks, gather_index

    def forward(self, sents, visual_feats):
        
        if self.tr_name.startswith("roberta"):
            input_ids, img_feats, img_pos_feats, attn_masks, gather_index = self.preprocess_roberta(sents, visual_feats, self.num_features, self.tokenizer)
        elif self.tr_name.startswith("bert"):
            input_ids, img_feats, img_pos_feats, attn_masks, gather_index = self.preprocess_bert(sents, visual_feats, self.num_features, self.tokenizer)

        seq_out, pooled_output = self.model(input_ids.cuda(), None, img_feats.cuda(), img_pos_feats.cuda(), attn_masks.cuda(), gather_index=gather_index.cuda())

        output = self.classifier(pooled_output)

        return output

    def get_gather_index(self, txt_lens, num_bbs, batch_size, max_len, out_size):

        assert len(txt_lens) == len(num_bbs) == batch_size
        gather_index = torch.arange(0, out_size, dtype=torch.long,
                                    ).unsqueeze(0).repeat(batch_size, 1)


        for i, (tl, nbb) in enumerate(zip(txt_lens, num_bbs)):
            # NOTE: SEQ_LEN + Num BBOXES MUST BE < MAX_SEQ LEN for this to work! Else non singleton dimension error! 
            gather_index.data[i, tl:tl+nbb] = torch.arange(max_len, max_len+nbb, dtype=torch.long).data

        return gather_index

    def save(self, path):
        torch.save(self.model.state_dict(),
                   os.path.join("%s_U.pth" % path))

    def load(self, path):
        # Load state_dict from snapshot file
        print("Load pre-trained model from %s" % path)
        state_dict = torch.load("%s" % path)
        new_state_dict = {}
        for key, value in state_dict.items():

            if 'uniter.' in key:
                key = key.replace('uniter.', '')

            # Unfortuantely their models are pretrained on bert-large-cased
            # Uncommenting the following will allow using an uncased model
            #if key.startswith("embeddings.word_embeddings.weight"):
            #    print("SKIPPING:", key)
            #    continue

            if key.startswith("img_embeddings.pos_linear.weight"):
                if args.num_pos == 6: # Loading all 7 (not 6) for UNITER
                    new_state_dict[key] = value
                else:
                    new_state_dict[key] = value[:, :4]
                    print("MODIFYING:", key)
            elif key.startswith("module."):
                new_state_dict[key[len("module."):]] = value
            # Masked pretrained model
            elif key.startswith("bert."):
                print("SAVING {} as {}.".format(key, key[5:]))
                new_state_dict[key[5:]] = value
            else:
                new_state_dict[key] = value

        state_dict = new_state_dict

        # Print out the differences of pre-trained and model weights.
        load_keys = set(state_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        print()
        print("Weights in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Weights in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()

        # Load weights to model
        self.model.load_state_dict(state_dict, strict=False)

    def init_weights(self, module):
        """ Initialize the weights """
        print("REINITING: ", module)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.model.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()