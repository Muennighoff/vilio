### TODO:
# Try adding our 4 pos features
# Try Add TOKTYPE IDS
# Try Set Use_img_layernormtotrue
# Try using transformers version as in theoriginal repo

# Reextract features using the open-sourced bottom up attention to get 2054? 

import os

import torch
import torch.nn as nn

from param import args

from src.vilio.modeling_bertX import BertLayerNorm, GeLU, BertLayer

from src.vilio.modeling_bertO import BertO
from src.vilio.transformers.tokenization_auto import AutoTokenizer

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

def preprocess_bert(sents, max_seq_len, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for sent in sents:
        # Remove double whitespaces
        sent = " ".join(str(sent).split())
        tokens = tokenizer.tokenize(sent)

        if len(tokens) > max_seq_len - 2:
            tokens = tokens[:(max_seq_len - 2)]
            print("Too long: ", tokens)

        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        segment_ids = [0] * len(input_ids)

        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_len - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        

        assert len(input_ids) == max_seq_len
        assert len(input_mask) == max_seq_len
        assert len(segment_ids) == max_seq_len

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
    return features

def preprocess_roberta(sents, max_seq_len, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for sent in sents:
        # Remove double whitespaces & append whitespace for Roberta
        sent = " " + " ".join(str(sent).split())
        tokens = tokenizer.tokenize(sent)

        # EXP --- 2 </s> as in Roberta
        if len(tokens) > max_seq_len - 3:
            tokens = tokens[:(max_seq_len - 2)]
            print("Too long: ", tokens)


        # Pair of sequences: <s> A </s></s> B </s>
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = [0] + input_ids + [2] + [2]

        segment_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)

        # Pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([1] * padding_length)
            input_mask = input_mask + ([0] * padding_length)
            segment_ids = segment_ids + ([0] * padding_length)

        assert len(input_ids) == max_seq_len
        assert len(input_mask) == max_seq_len
        assert len(segment_ids) == max_seq_len

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids))
    return features


class ModelO(nn.Module):
    """
    Oscar Model with varying Bert Encoders.
    """
    def __init__(self, args=args, max_seq_len=128, max_img_seq_len=args.num_features, tr_name=args.tr):
        """
        max_seq_len: Or Repo - VQA: 128
        max_img_seq_len: Or Repo - NLVR2: 40 // GQA: 45 // VQA: 50 --- Set to args.num_features, as we dont have padding implemented
        tr_name: transformer model
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tr_name = tr_name
        self.max_img_seq_len = max_img_seq_len

        ### BUILD TOKENIZER ###
        self.tokenizer = AutoTokenizer.from_pretrained(tr_name)

        ### BUILD MODEL ###
        if tr_name.startswith("bert"):
            self.model, loading_info = BertO.from_pretrained(tr_name, output_loading_info=True, 
                                                            img_feature_dim=2048 + args.num_pos)

        print("UNEXPECTED: ", loading_info["unexpected_keys"])
        print("MISSING: ", loading_info["missing_keys"])
        print("ERRORS: ", loading_info["error_msgs"])


        ### CLASSIFICATION HEADS ###
        # LXRT Default classifier tends to perform best; For Albert gelu_new outperforms gelu
        # Make sure to only have used stuff below as it seems to have an effect on random initilization!

        if args.reg:
            self.high_dropout = nn.Dropout(p=0.5)
            self.classifier = nn.Linear(self.model.config.hidden_size, 2)
            self.model.pooler.apply(self.init_weights)
        else:
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

    def forward(self, sents, visual_feats, visual_attention_mask=None):
        
        if self.tr_name.startswith("bert") or self.tr_name.startswith("albert"):
            train_features = preprocess_bert(sents, self.max_seq_len, self.tokenizer)

        input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).cuda()
        input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long).cuda()

        img_feat, img_pos_feat = visual_feats

        # Cat Pos feats into img feats
        img_feat = torch.cat((img_feat, img_pos_feat), dim = -1).cuda()
        # They only use 50 feats in or repo
        img_feat = img_feat[:, :self.max_img_seq_len]

        image_mask = torch.ones((input_ids.shape[0], self.max_img_seq_len), dtype=torch.long)
        input_mask = torch.cat((input_mask, image_mask), dim = -1).cuda()

        seq_out, output = self.model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, img_feats=img_feat)

        if args.reg:
            # multisample dropout (wut): https://arxiv.org/abs/1905.09788
            output = torch.mean(
                torch.stack(
                    [self.classifier(self.high_dropout(seq_out)) for _ in range(5)],
                    dim=0,
                ),
                dim=0,
            )

        else:
            output = self.classifier(output)

        return output

    def save(self, path):
        torch.save(self.model.state_dict(),
                   os.path.join("%s_O.pth" % path))

    def load(self, path):
        # Load state_dict from snapshot file
        print("Load pre-trained model from %s" % path)
        state_dict = torch.load("%s" % path)
        new_state_dict = {}
        for key, value in state_dict.items():
            
            if key.startswith("bert.img_embedding.weight"):
                if args.num_pos == 6:
                    new_state_dict[key[5:]] = value
                else:
                    value = value[:, :2052].clone()
                    new_state_dict[key[5:]] = value
                    print("MODIFYING:", key)

            # Masked pretrained model
            elif key.startswith("bert."):
                print("SAVING {} as {}.".format(key, key[5:]))
                new_state_dict[key[5:]] = value
            
            elif key.startswith("module."):
                new_state_dict[key[len("module."):]] = value
          
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

    def reinit_weights(self, module):
        """ Re-init final bert weights for a better model """

        # This refers to the LXRTEncoder from modeling
        if isinstance(module, nn.ModuleList):
            if isinstance(module[-1], BertLayer):
                print("Reiniting :", module[-1])
                # Reinit that layer: 
                module[-2:].apply(self.init_weights)
        # Alternatively for child in module.children() can be used
