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
from ..layer import Encoder, Decoder, Embedding, Linear, RelativePositionEmbedding
from .config import CPM2Config
from .basemodel import BaseModel

class CPM2(BaseModel):
    _CONFIG_TYPE = CPM2Config
    
    def __init__(self, config: CPM2Config):
        
        super().__init__()

        self.encoder = Encoder(
            num_layers = config.num_encoder_layers,
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
        )

        self.decoder = Decoder(
            num_layers = config.num_decoder_layers,
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
        )

        self.input_embedding = Embedding(
            vocab_size = config.vocab_size, 
            embedding_size = config.dim_model,
            length_scale = False, # TODO not an elegent implementation # config.length_scale,
            dtype = config.dtype,
            int8 = config.int8,
            init_mean = config.emb_init_mean,
            init_std = config.emb_init_std,
        )

        self.position_bias_enc = RelativePositionEmbedding(
            num_heads = config.num_heads, 
            num_buckets = config.position_bias_num_buckets, 
            max_distance = config.position_bias_max_distance, 
            bidirectional = True, 
            dtype = config.dtype,
            init_mean = config.pos_init_mean,
            init_std = config.pos_init_std,
        )

        self.position_bias_dec = RelativePositionEmbedding(
            num_heads = config.num_heads, 
            num_buckets = config.position_bias_num_buckets, 
            max_distance = config.position_bias_max_distance, 
            bidirectional = False, 
            dtype = config.dtype,
            init_mean = config.pos_init_mean,
            init_std = config.pos_init_std,
        )

        self.cls_head = config.cls_head
        self.output_projection = Linear(
            dim_out = self.cls_head if self.cls_head else config.vocab_size,
            dim_in = config.dim_model,
            length_scale = config.length_scale,
            dtype = config.dtype,
            int8 = config.int8,
            init_mean = config.proj_init_mean,
            init_std = config.proj_init_std,
            bias = config.proj_bias,
        )

    def forward(self, 
                enc_input : torch.Tensor, # (batch, seq_enc)
                enc_length : torch.Tensor, # (batch)
                dec_input : torch.Tensor, # (batch, seq_dec)
                dec_length : torch.Tensor, # (batch)
        ):
        """ This model inherits from BaseModel. This model is also a PyTorch torch.nn.Module subclass.
            You can use it as a regular PyTorch Module.

        Args:
            enc_input (:obj:`torch.Tensor` of shape ``(batch, seq_enc)``): Indices of input sequence tokens for encoder. It will be embedded by model's internal embedding lookup matrix.
            enc_length (:obj:`torch.Tensor` of shape ``(batch)``): Length of input sequence for encoder before padding.  
            dec_input (:obj:`torch.Tensor` of shape ``(batch, seq_dec)``): Indices of input sequence tokens for decoder. It will be embedded by model's internal embedding lookup matrix.
            dec_length (:obj:`torch.Tensor` of shape ``(batch)``): Length of input sequence for encoder before padding.

        Return:
            torch.Tensor of shape (batch, seq_dec, vocab_output_size) or (batch, seqlen, cls_head): The CPM-2 output. Prediction scores of the language modeling before SoftMax.
        """
        
        batch = enc_input.size(0)
        seq_enc = enc_input.size(1)
        seq_dec = dec_input.size(1)

        with torch.no_grad():

            device = enc_input.device

            enc_mask_1d = torch.arange(seq_enc, device=device)[None, :].repeat(batch, 1) < enc_length[:, None]
            dec_mask_1d = torch.arange(seq_dec, device=device)[None, :].repeat(batch, 1) < dec_length[:, None]
            directional_mask_2d = torch.arange(seq_dec, device=device) <= torch.arange(seq_dec, device=device).view(-1, 1)
            # (batch, seq_enc, seq_enc)
            enc_attention_mask = enc_mask_1d.view(batch, seq_enc, 1) & enc_mask_1d.view(batch, 1, seq_enc)
            # (batch, seq_dec, seq_dec)
            dec_attention_mask = dec_mask_1d.view(batch, seq_dec, 1) & dec_mask_1d.view(batch, 1, seq_dec) & directional_mask_2d.view(1, seq_dec, seq_dec)
            # (batch, seq_dec, seq_enc)
            cross_attention_mask = enc_mask_1d.view(batch, 1, seq_enc) & dec_mask_1d.view(batch, seq_dec, 1)

        # (num_heads, seq_enc, seq_enc)
        enc_position_bias = self.position_bias_enc(seq_enc, seq_enc)
        # (num_heads, seq_dec, seq_dec)
        dec_position_bias = self.position_bias_dec(seq_dec, seq_dec)

        # (batch, seq_enc, dim_model)
        hidden_states_enc = self.input_embedding(enc_input)
        # (batch, seq_enc, dim_model)
        hidden_states_enc = self.encoder(hidden_states_enc, enc_attention_mask, enc_position_bias)

        # (batch, seq_dec, dim_model)
        hidden_states_dec = self.input_embedding(dec_input)
        # (batch, seq_dec, dim_model)
        hidden_states_dec = self.decoder(hidden_states_dec, dec_attention_mask, dec_position_bias,
                                         hidden_states_enc, cross_attention_mask, None)

        # (batch, seq_dec, vocab_output_size)
        logits = self.output_projection(hidden_states_dec)
        return logits

