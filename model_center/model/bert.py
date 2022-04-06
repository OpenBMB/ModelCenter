# coding=utf-8
# Copyright 2022 The OpenBMB team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch

from ..layer import Encoder, Embedding, Linear, LayerNorm
from .basemodel import BaseModel
from .config import BertConfig
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

class BertPooler(torch.nn.Module):
    def __init__(self, dim_model):
        super().__init__()
        self.dense = Linear(dim_model, dim_model, bias=True)
        self.activation = torch.nn.Tanh()

    def forward(self, hidden_states):
        pooled_output = self.dense(hidden_states[:, 0, :])
        pooled_output = self.activation(pooled_output)
        return pooled_output

        
class BertLMHead(torch.nn.Module):
    def __init__(self, dim_model, vocab_size, norm_eps):
        super().__init__()
        self.dense = Linear(dim_model, dim_model, bias=True)
        self.act_fn = torch.nn.functional.gelu
        self.layer_norm = LayerNorm(dim_model, eps=norm_eps)
        self.decoder = Linear(dim_model, vocab_size, bias=True)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        logits = self.decoder(hidden_states)
        return logits


class Bert(BaseModel):

    _CONFIG_TYPE = BertConfig

    def __init__(self, config: BertConfig):
        super().__init__()

        self.input_embedding = Embedding(
            vocab_size = config.vocab_size,
            embedding_size = config.dim_model,
            length_scale = config.length_scale,
            dtype = config.dtype,
            int8 = config.int8,
            init_mean = config.emb_init_mean,
            init_std = config.emb_init_std,
        )

        self.position_embedding = Embedding(
            vocab_size = config.position_size, 
            embedding_size = config.dim_model,
            length_scale = config.length_scale,
            dtype = config.dtype,
            int8 = config.int8,
            init_mean = config.emb_init_mean,
            init_std = config.emb_init_std,
        )

        self.token_type_embedding = Embedding(
            vocab_size = config.type_size,
            embedding_size = config.dim_model,
            length_scale = config.length_scale,
            dtype = config.dtype,
            int8 = config.int8,
            init_mean = config.emb_init_mean,
            init_std = config.emb_init_std,
        )

        self.embed_dropout = torch.nn.Dropout(config.dropout_p)

        self.encoder = Encoder(
            num_layers = config.num_layers,
            dim_model = config.dim_model, 
            dim_ff = config.dim_ff,
            num_heads = config.num_heads,
            dim_head = config.dim_head,
            dtype = config.dtype, 
            int8 = config.int8,
            norm_eps = config.norm_eps, 
            norm_init_var = config.norm_init_var,
            norm_bias = config.norm_bias,
            att_init_mean = config.att_init_mean, 
            att_init_std = config.att_init_std,
            att_bias = config.att_bias,
            att_mask_value = float(config.att_mask_value),
            pos_bias_type = config.pos_bias_type,
            ffn_init_mean = config.ffn_init_mean, 
            ffn_init_std = config.ffn_init_std,
            ffn_bias = config.ffn_bias,
            ffn_activate_fn = config.ffn_activate_fn,
            length_scale = config.length_scale,
            attn_scale = config.attn_scale,
            dropout_p = config.dropout_p,
            post_layer_norm = config.post_layer_norm,
        )

        self.tied = config.tied
        self.cls_head = config.cls_head
        if self.cls_head:
            self.cls_projection = Linear(
                dim_out = self.cls_head,
                dim_in = config.dim_model,
                length_scale = config.length_scale,
                dtype = config.dtype,
                int8 = config.int8,
                init_mean = config.proj_init_mean,
                init_std = config.proj_init_std,
                bias = config.proj_bias,
            )
        if not self.tied:
            self.lm_head = BertLMHead(
                dim_model = config.dim_model,
                vocab_size = config.vocab_size,
                norm_eps = config.norm_eps,
            )

        self.pooler = BertPooler(config.dim_model)

    def forward(self,
                input_ids=None,
                length=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None, #unused
                inputs_embeds=None,
                encoder_hidden_states=None, #unused
                encoder_attention_mask=None, #unused
                output_attentions=None, #unused
                output_hidden_states=None, #unused
                return_dict=True,
                return_logits = False,
    ):
        """ This model inherits from BaseModel. This model is also a PyTorch torch.nn.Module subclass.
            You can use it as a regular PyTorch Module.
            You can also select the data and data type that you want the model to return through changing the value of `return_dict` and `return_logits`.

        Args:
            input_ids (:obj:`torch.Tensor` of shape ``(batch, seq_length)``): Indices of input sequence tokens. It will be embedded by model's internal embedding lookup matrix.
            length (:obj:`torch.Tensor` of shape ``(batch)``): Length of input sequence before padding.  
            attention_mask (:obj:`torch.Tensor` of shape ``(batch, seq_length)``): Used to avoid performing attention on padding token indices.
            token_type_ids(:obj:`torch.Tensor` of shape ``(batch, seq_length)``): Unused. 
            position_ids(:obj:`torch.Tensor` of shape ``(batch, seq_length)``): Unused.
            head_mask (:obj:`torch.Tensor` of shape ``(num_layers, num_heads)``): Unused.
            inputs_embeds (:obj:`torch.Tensor` of shape ``(batch, seq_length, dim_model)``): Embedding of the input. You can choose to directly pass the inputs embedding to control the way of embedding. 
            encoder_hidden_states(:obj:`torch.Tensor` of shape(batch, seq_length, dim_model)): Unused.
            encoder_attention_mask (:obj:`torch.Tensor` of shape ``(batch, seq_length)``): Unused. 
            output_attentions (:obj:`torch.Tensor` of shape ``(batch, num_heads, seq_length, seq_length)``): Unused.
            output_hidden_states (:obj:`torch.Tensor` of shape ``(batch, seq_length, dim_model)``): Unused.
            return_dict (:obj:`bool`): Whether to return a BaseModelOutputWithPoolingAndCrossAttentions instead of just a tuple.
            return_logits (:obj:`bool`): Whether to return the prediction score for each token in vocabulary (before softmax).

        Return:
            BaseModelOutputWithPoolingAndCrossAttentions or tuple or torch.Tensor of shape (batch, seq_length, vocab_output_size) or (batch, seqlen, cls_head): The Bert output. Depended on the value of `return_dict` and `return_logits` 

        """
        assert input_ids is not None or inputs_embeds is not None

        if input_ids is not None:
            batch = input_ids.size(0)
            seq_length = input_ids.size(1)
            device = input_ids.device
        else:
            batch = inputs_embeds.size(0)
            seq_length = inputs_embeds.size(1)
            device = inputs_embeds.device

        with torch.no_grad():

            if attention_mask is not None:
                attention_mask = attention_mask.to(torch.bool)
            else:
                attention_mask = torch.arange(seq_length, device=device)[None, :].repeat(batch, 1) < length[:, None]
            attention_mask = attention_mask.view(batch, seq_length, 1) & attention_mask.view(batch, 1, seq_length)

            if position_ids is None:
                position_ids = torch.arange(seq_length, dtype=torch.int32, device=device)[None, :].repeat(batch, 1)

            if token_type_ids is None:
                token_type_ids = torch.zeros(seq_length, dtype=torch.int32, device=device)[None, :].repeat(batch, 1)

        if inputs_embeds is None:
            hidden_states = self.input_embedding(input_ids.to(torch.int32))
        else:
            hidden_states = inputs_embeds
        position_embeds = self.position_embedding(position_ids.to(torch.int32))
        token_type_embeds = self.token_type_embedding(token_type_ids.to(torch.int32))
        hidden_states = hidden_states + token_type_embeds + position_embeds

        hidden_states = self.embed_dropout(hidden_states)

        hidden_states = self.encoder(hidden_states, attention_mask)

        if self.cls_head:
            logits = self.cls_projection(hidden_states)
        elif self.tied:
            logits = self.input_embedding.projection(hidden_states)
        elif not self.tied:
            logits = self.lm_head(hidden_states)

        if return_logits:
            return logits

        pooled_output = self.pooler(hidden_states)

        if not return_dict:
            return (hidden_states, pooled_output, None, None, None, None)
        else:
            return BaseModelOutputWithPoolingAndCrossAttentions(
                last_hidden_state=hidden_states,
                pooler_output=pooled_output,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
                cross_attentions=None,
            )