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
from ..layer import Encoder, Embedding, Linear, RelativePositionEmbedding
from .basemodel import BaseModel
from .config import CPM1Config
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions


class CPM1(BaseModel):

    _CONFIG_TYPE = CPM1Config
    
    def __init__(self, config: CPM1Config):
        
        super().__init__()

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
            use_cache = config.use_cache             
        )

        self.input_embedding = Embedding(
            vocab_size = config.vocab_size, 
            embedding_size = config.dim_model,
            length_scale = config.length_scale,
            dtype = config.dtype,
            int8 = config.int8,
            init_mean = config.emb_init_mean,
            init_std = config.emb_init_std,
        )

        self.position_bias = RelativePositionEmbedding(
            num_heads = config.num_heads, 
            num_buckets = config.position_bias_num_buckets, 
            max_distance = config.position_bias_max_distance, 
            bidirectional = True,
            dtype = config.dtype,
            init_mean = config.pos_init_mean,
            init_std = config.pos_init_std,
        )
        
        self.tied = config.tied
        self.cls_head = config.cls_head
        if self.cls_head:
            self.output_projection = Linear(
                dim_out = config.cls_head,
                dim_in = config.dim_model,
                length_scale = config.length_scale,
                dtype = config.dtype,
                int8 = config.int8,
                init_mean = config.proj_init_mean,
                init_std = config.proj_init_std,
                bias = config.proj_bias,
            )
        elif not self.tied:
            self.output_projection = Linear(
                dim_out = config.vocab_size,
                dim_in = config.dim_model,
                length_scale = config.length_scale,
                dtype = config.dtype,
                int8 = config.int8,
                init_mean = config.proj_init_mean,
                init_std = config.proj_init_std,
                bias = config.proj_bias,
            )

    def forward(self, input_ids : torch.Tensor, # (batch, seqlen)
                      length : torch.Tensor, # (batch)
                      context : torch.Tensor, # (batch, seqlen)
                      span : torch.Tensor, # (batch, seqlen)
                      inputs_embeds = None, # (batch, seqlen, dim)
                      encoder_hidden_states = None, #unused
                      encoder_attention_mask = None, #unused
                      use_cache=False,
                      past_key_values=None,
                      output_attentions = None, #unused
                      output_hidden_states = None, #unused
                      return_dict = True,
                      return_logits = False,
        ): # (batch, seqlen)
        """ This model inherits from BaseModel. This model is also a PyTorch torch.nn.Module subclass.
            You can use it as a regular PyTorch Module.
            
        Args:
            input (:obj:`torch.Tensor` of shape ``(batch, seqlen)``): 
            length (:obj:`torch.Tensor` of shape ``(batch)``): 
            context (:obj:`torch.Tensor` of shape ``(batch, seqlen)``): 
            span (:obj:`torch.Tensor` of shape ``(batch, seqlen)``): 
        Return:
            torch.Tensor of shape (batch, seqlen, vocab_size) or (batch, seqlen, cls_head): The CPM output. Prediction scores of the language modeling before SoftMax.

        """
        assert input_ids is not None or inputs_embeds is not None

        if input_ids is not None:
            batch = input_ids.size(0)
            input_length = input_ids.size(1)
            device = input_ids.device
        else:
            batch = inputs_embeds.size(0)
            input_length = inputs_embeds.size(1)
            device = inputs_embeds.device

        pkv_len = 0 if past_key_values is None else past_key_values[0][0].size(-2)
        seq_length = pkv_len + input_length

        with torch.no_grad():

            directional_mask_2d = torch.arange(seq_length, device=device) <= torch.arange(seq_length, device=device).view(-1, 1)
            attention_mask = context[:, None, :] | (context[:, :, None].logical_not() & directional_mask_2d.view(1, seq_length, seq_length))
            attention_mask = attention_mask & (span[:, None, :] == span[:, :, None])

            mask_1d = torch.arange(seq_length, device=device)[None, :].repeat(batch, 1) < length[:, None]
            attention_mask = mask_1d.view(batch, seq_length, 1) & mask_1d.view(batch, 1, seq_length) & attention_mask

            position_bias = self.position_bias(seq_length, seq_length)

        attention_mask = attention_mask[:, -input_length:, :]
        position_bias = position_bias[:, -input_length:, :]
        hidden_states = self.input_embedding(input)

        current_key_values = None
        if use_cache:
            hidden_states, current_key_values = self.encoder(hidden_states, attention_mask, position_bias,
                                                             use_cache = use_cache, past_key_values = past_key_values)
        else:
            hidden_states = self.encoder(hidden_states, attention_mask)

        if self.cls_head:
            logits = self.output_projection(hidden_states)
        elif not self.tied:
            logits = self.output_projection(hidden_states)
        else:
            logits = self.input_embedding.projection(hidden_states)

        if return_logits:
            return logits

        if not return_dict:
            return tuple(hidden_states, None, None, None, None)
        else:
            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=current_key_values,
                hidden_states=None,
                attentions=None,
                cross_attentions=None,
            )
