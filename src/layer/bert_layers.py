import math
import inspect
from dataclasses import dataclass
from model.config.bert_config import BertConfig
from typing import Optional, Tuple, Callable

import torch
from torch import nn
import bmpretrain as bmp

from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from layer import Embedding, LayerNorm, Linear, SelfAttentionBlock, TransformerBlock


class BertEmbeddings(torch.nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = Embedding(config.type_vocab_size, config.hidden_size)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.register_buffer("token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False,)

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]


        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            buffered_token_type_ids = self.token_type_ids[:, :seq_length]
            buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
            token_type_ids = buffered_token_type_ids_expanded

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids.to(torch.int32))
        token_type_embeddings = self.token_type_embeddings(token_type_ids.to(torch.int32))

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids.to(torch.int32))
            embeddings += position_embeddings

        return embeddings


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = Linear(config.hidden_size, config.hidden_size, bias=True)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, :, 0:1].repeat(1,1,2).contiguous()
        pooled_output = self.dense(first_token_tensor)[:, :, 0]
        pooled_output = self.activation(pooled_output)
        return pooled_output
    

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.intermediate_act_fn = nn.functional.gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = hidden_states + input_tensor

        hidden_states = hidden_states
        return hidden_states

class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = SelfAttentionBlock(
            config.hidden_size,
            config.num_attention_heads,
            int(config.hidden_size / config.num_attention_heads),
            dropout_p=config.attention_probs_dropout_prob,
            attn_scale=True,
            att_bias=True,
            norm_bias=True,
            norm_eps=config.layer_norm_eps,
            post_layer_norm=True,
        )
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        attention_output = self.attention(
            hidden_states,
            attention_mask,
        )

        attention_output = self.LayerNorm(attention_output)

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config : BertConfig):
        super().__init__()
        self.config = config
        #self.layer = bmp.TransformerBlockList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.layer = nn.ModuleList([
            TransformerBlock(
                dim_model=config.hidden_size,
                dim_ff=config.intermediate_size,
                num_heads=config.num_attention_heads,
                dim_head=int(config.hidden_size / config.num_attention_heads),
                dropout_p=config.attention_probs_dropout_prob,
                attn_scale=True,
                att_bias=True,
                norm_bias=True,
                norm_eps=config.layer_norm_eps,
                post_layer_norm=True,
                ffn_activate_fn="gelu",
                ffn_bias=True,
            ) for _ in range(config.num_hidden_layers)
        ])
        self.LayerNorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        return_dict=True,
    ):
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(
                hidden_states,
                attention_mask,
            )
        
        hidden_states = self.LayerNorm(hidden_states)

        if not return_dict:
            return (hidden_states, )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )