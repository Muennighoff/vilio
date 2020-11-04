"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.
Pytorch modules
some classes are modified from HuggingFace
(https://github.com/huggingface/transformers)
"""
import copy
import json
import logging
from io import open
import math


import torch
from torch import nn
# They used apex's FusedLayerNorm in or. repo --- In case users would like to use FusedLayerNorm, import the original one from apex
from src.vilio.transformers.modeling_bert import BertLayerNorm as FusedLayerNorm
from src.vilio.transformers.modeling_bert import BertLayerNorm, BertPreTrainedModel
from src.vilio.transformers.configuration_bert import BertConfig

import math
import torch
import numbers
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss, SmoothL1Loss

import importlib

from param import args

### BOTTOM-UP APPROACH ###
# We start with the lowest level & then go up step by step

# A) ACTIVATION FUNCS & LOGGER
# B) CONFIGS / HYPERPARAMS FOR VISION & LANG
# C) BERT/ATTENTION HELPER FUNCS
# D) PRE_TRAINED MODEL / PARENT CLASS
# E) EMBEDDING ENCODERS FOR VISION & LANG
# F) ENCODER & LAYERS
# G) FINAL UNITER MODEL
# H) PRETRAINING

### A) ACTIVATION FUNCS ###

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

logger = logging.getLogger(__name__)


### B) CONFIGS ###
# Note: We removed the original UniterConfig for a simple BertConfig as imported

### C) BERT HELPER FUNCS ###

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

### D) PRETRAINED MODEL ###
# We removed the original UniterPreTrainedModel in favor of BertPreTrainedModel as imported

### E) EMBEDDINGS FOR LANG & VISION ###

# This is essentially the same as BertEmbeddings except for padding_idx - I made necessary updates to reflect BertEmbeddings as of 08/2020
class UniterTextEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.hidden_size, padding_idx=0)
        #self.word_embeddings = nn.Embedding(28996,config.hidden_size, padding_idx=0)


        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)

        # NOTE: type_vocab_size is 2 for BERT, but 1 for ROBERTA!
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                  config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model
        # variable name and be able to load any TensorFlow checkpoint file
        self.LayerNorm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids, position_ids=None, token_type_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # NOTE: For some reason the p. embeddings are not scaled to the size they should have / (A pytorch upgrade regarding Broadcasting?)
        # Hence we repeat them by BS to make the addition afterwards possible
        position_embeddings = position_embeddings.repeat(token_type_embeddings.size(0), 1, 1)

        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class UniterImageEmbeddings(nn.Module):
    def __init__(self, config, img_dim):
        super().__init__()
        self.img_linear = nn.Linear(img_dim, config.hidden_size)
        self.img_layer_norm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.pos_layer_norm = FusedLayerNorm(config.hidden_size, eps=1e-12)

        if args.num_pos == 6:
            # Use the full 7 pos as in original
            self.pos_linear = nn.Linear(7, config.hidden_size)
        else:
            self.pos_linear = nn.Linear(4, config.hidden_size)

        # tf naming convention for layer norm
        self.LayerNorm = FusedLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, img_feat, img_pos_feat, type_embeddings):
        transformed_im = self.img_layer_norm(self.img_linear(img_feat))
        transformed_pos = self.pos_layer_norm(self.pos_linear(img_pos_feat))
        embeddings = transformed_im + transformed_pos + type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings

### F) ENCODER & LAYERS ###

class UniterEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, input_, attention_mask,
                output_all_encoded_layers=True):
        all_encoder_layers = []
        hidden_states = input_
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

### G) FINAL MODEL ###

class BertU(BertPreTrainedModel):
    """ Modification for Joint Vision-Language Encoding
    """
    # Note: This allows loading in bert weights & the bert config (which is the ~same as the uniterconfig)
    config_class = BertConfig

    def __init__(self, config, img_dim=2048):
        super().__init__(config)
        self.embeddings = UniterTextEmbeddings(config)
        self.img_embeddings = UniterImageEmbeddings(config, img_dim)
        self.encoder = UniterEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def _compute_txt_embeddings(self, input_ids, position_ids,
                                txt_type_ids=None):
        output = self.embeddings(input_ids, position_ids, txt_type_ids)
        return output

    def _compute_img_embeddings(self, img_feat, img_pos_feat,
                                img_type_ids=None):

        if img_type_ids is None:
            img_type_ids = torch.ones_like(img_feat[:, :, 0].long())

        img_type_embeddings = self.embeddings.token_type_embeddings(
            img_type_ids)

        output = self.img_embeddings(img_feat, img_pos_feat,
                                     img_type_embeddings)
        return output

    def _compute_img_txt_embeddings(self, input_ids, position_ids,
                                    img_feat, img_pos_feat,
                                    gather_index,
                                    txt_type_ids=None, img_type_ids=None):
        """
        NOTES FROM ME:
        The idea here is to incorporate our image embeddings into the existing text embeddings!
        > Our text embeddings are of shape BS, MAX_SEQ_LEN PER BATCH, HIDDEN_STATES (e.g. 32, 128, 768)
        > The IMG embedding has a shape of BS, BBOXES, HIDDEN_STATES (e.g. 32, 36, 768)
        > Via GATHER we replace unused tokens & place the IMG embedding at the end

        Uncomment the print & if statements to see it happening in action!
        """
        txt_emb = self._compute_txt_embeddings(
            input_ids, position_ids, txt_type_ids)
        img_emb = self._compute_img_embeddings(
            img_feat, img_pos_feat, img_type_ids)

        #print(" EMBEDDING SHAPES : ", txt_emb.shape, img_emb.shape)

        # Check the first 10 input ids first hidden stete
        #print("TST-EMB: ", txt_emb[0, :40, 0])
        #print("IMG-EMB: ", img_emb[0, :40, 0])

        # align back to most compact input
        gather_index = gather_index.unsqueeze(-1).expand(
            -1, -1, self.config.hidden_size)
        embedding_output = torch.gather(torch.cat([txt_emb, img_emb], dim=1),
                                        dim=1, index=gather_index)

        #print(" EMB OUTPUT SHAPE : ", embedding_output.shape)
        #print(" EMB OUTPUT : ", embedding_output[0, :40, 0])

        #if torch.all(torch.eq(txt_emb, embedding_output)):
        #    print("SAME!")
        #else:
        #    diff = txt_emb - embedding_output
        #    print("DIFF: ", diff[0, :, 0])

        return embedding_output

    # Note: output_all_encoded_layers is set to TRUE in the original repo
    def forward(self, input_ids, position_ids,
                img_feat, img_pos_feat,
                attention_mask, gather_index=None,
                output_all_encoded_layers=False,
                txt_type_ids=None, img_type_ids=None):

        #print("ATTN:", attention_mask.shape)
        #print("ATTN: ", attention_mask[0, :140])

        # compute self-attention mask
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # embedding layer
        if input_ids is None:
            # image only
            embedding_output = self._compute_img_embeddings(
                img_feat, img_pos_feat, img_type_ids)
        elif img_feat is None:
            # text only
            embedding_output = self._compute_txt_embeddings(
                input_ids, position_ids, txt_type_ids)
        else:
            embedding_output = self._compute_img_txt_embeddings(
                input_ids, position_ids,
                img_feat, img_pos_feat,
                gather_index, txt_type_ids, img_type_ids)

        encoded_layers = self.encoder(
            embedding_output, extended_attention_mask,
            output_all_encoded_layers=output_all_encoded_layers)


        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]

        # Applying the pooler -- In or. repo they apply it in various submodels; We apply it here for consistency with or. transformer models
        pooled_output = self.pooler(encoded_layers)

        return encoded_layers, pooled_output

### H) PRETRAINING ONLY ###

### ACTIVATION FUNCS ###

from src.vilio.transformers.activations import gelu, gelu_new, swish

def mish(x):
    return x * torch.tanh(nn.functional.softplus(x))

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new, "mish": mish}

class GeLU(nn.Module):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


class VisualConfig(object):
    VISUAL_LOSSES = ['obj', 'attr', 'feat']
    def __init__(self,
                 l_layers=12,
                 x_layers=5,
                 r_layers=0):
        self.l_layers = l_layers
        self.x_layers = x_layers
        self.r_layers = r_layers

        self.visual_feat_dim = 2048
        self.visual_pos_dim = 4

        self.obj_id_num = 1600
        self.attr_id_num = 400

        self.visual_losses = self.VISUAL_LOSSES
        self.visual_loss_config = {
            'obj': (self.obj_id_num, 'ce', (-1,), 1/0.15),
            'attr': (self.attr_id_num, 'ce', (-1,), 1/0.15),
            'feat': (2048, 'l2', (-1, 2048), 1/0.15),
        }

    def set_visual_dims(self, feat_dim, pos_dim):
        self.visual_feat_dim = feat_dim
        self.visual_pos_dim = pos_dim

VISUAL_CONFIG = VisualConfig()

class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states

class BertVisualObjHead(nn.Module):
    def __init__(self, config, visual_losses):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # Decide the use of visual losses
        visual_losses = visual_losses.split(",")
        for loss in visual_losses:
            assert loss in VISUAL_CONFIG.VISUAL_LOSSES
        self.visual_losses = visual_losses

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder_dict = nn.ModuleDict({
            key: nn.Linear(config.hidden_size, VISUAL_CONFIG.visual_loss_config[key][0])
            for key in self.visual_losses
        })

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        output = {}
        for key in self.visual_losses:
            output[key] = self.decoder_dict[key](hidden_states)
        return output


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score

class BertUPretraining(BertPreTrainedModel):
    def __init__(self,
                 config,
                 task_mask_lm=True,
                 task_matched=True,
                 task_obj_predict=True,
                 visual_losses='',
                 task_qa=False,
                 num_answers=2):
        super().__init__(config)
        # Configuration
        self.config = config
        self.num_answers = num_answers

        # Use of pre-training tasks
        self.task_mask_lm = task_mask_lm
        self.task_obj_predict = task_obj_predict
        self.task_matched = task_matched
        self.task_qa = task_qa

        # LXRT backbone
        self.bert = BertU(config)

        # Pre-training heads
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        if self.task_obj_predict:
            self.obj_predict_head = BertVisualObjHead(config, visual_losses)

        # Weight initialization
        self.init_weights()


    def forward(self, input_ids, position_ids, img_feats, img_pos_feats, attn_masks, gather_index, masked_lm_labels=None, 
                obj_labels=None, matched_label=None, ans=None):
    
        sequence_output, pooled_output = self.bert(
            input_ids, None, img_feats, img_pos_feats, attn_masks, gather_index
        )

        lang_prediction_scores, cross_relationship_score = self.cls(sequence_output, pooled_output)

        if self.task_qa:
            answer_score = self.answer_head(pooled_output)
        else:
            # This answer_score would not be used anywhere,
            # just to keep a constant return function signature.
            answer_score = pooled_output[0][0]

        total_loss = 0.
        loss_fct = CrossEntropyLoss(ignore_index=-1)
        losses = ()
        if masked_lm_labels is not None and self.task_mask_lm:
            masked_lm_loss = loss_fct(
                lang_prediction_scores.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1)
            )
            total_loss += masked_lm_loss
            losses += (masked_lm_loss.detach(),)

        if matched_label is not None and self.task_matched:
            matched_loss = loss_fct(
                cross_relationship_score.view(-1, 2),
                matched_label.view(-1)
            )
            total_loss += matched_loss
            losses += (matched_loss.detach(),)


        return total_loss, torch.stack(losses).unsqueeze(0), answer_score.detach()