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
import cpm_kernels.torch as ct
import bmtrain as bmt

from .blocks import TransformerBlock
from .layernorm import LayerNorm


class Encoder(torch.nn.Module):
    def __init__(self, 
            num_layers : int,
            dim_model : int, 
            dim_ff : int,
            num_heads : int,
            dim_head : int,
            dtype : torch.dtype = torch.half,
            int8 : bool = False, 
            norm_init_var = 1.0,
            norm_bias = False,
            norm_eps : float = 1e-5, 
            att_init_mean = 0.0, 
            att_init_std = 0.02,
            att_bias = False,
            att_mask_value = float("-inf"),
            ffn_init_mean = 0.0, 
            ffn_init_std = 0.02,
            ffn_bias = False,
            ffn_activate_fn = "gated_gelu",
            pos_bias_type = "none",
            post_layer_norm = False,
            length_scale : bool = False,
            attn_scale : bool = False,
            dropout_p = 0,
            parallel_ffn = False,
        ):

        super().__init__()
        
        self.num_layers = num_layers

        self.layers = bmt.TransformerBlockList([
            bmt.CheckpointBlock(
                TransformerBlock(
                    dim_model = dim_model, 
                    dim_ff = dim_ff,
                    num_heads = num_heads,
                    dim_head = dim_head,
                    is_decoder = False,
                    dtype = dtype, 
                    int8 = int8,
                    norm_eps = norm_eps, 
                    norm_init_var = norm_init_var,
                    norm_bias = norm_bias,
                    att_init_mean = att_init_mean, 
                    att_init_std = att_init_std,
                    att_bias = att_bias,
                    att_mask_value = att_mask_value,
                    ffn_init_mean = ffn_init_mean, 
                    ffn_init_std = ffn_init_std,
                    ffn_bias = ffn_bias,
                    ffn_activate_fn = ffn_activate_fn,
                    pos_bias_type = pos_bias_type,
                    post_layer_norm = post_layer_norm,
                    length_scale = length_scale,
                    attn_scale = attn_scale,
                    dropout_p = dropout_p,
                    parallel_ffn = parallel_ffn,
                )
            )
            for _ in range(num_layers)
        ])

        self.output_layernorm = LayerNorm(
                    dim_norm = dim_model, 
                    bias = norm_bias, 
                    dtype = dtype,
                    eps = norm_eps,
                    init_var = norm_init_var)

    def forward(self, hidden_states : torch.Tensor,     # (batch, dim_model, seq_enc)
                      attention_mask : torch.Tensor,    # (batch, seq_enc, seq_enc)
                      position_bias : torch.Tensor = None,     # (num_heads, seq_enc, seq_enc)
                      ):
        """ This class is a PyTorch torch.nn.Module subclass.You can use it as a regular PyTorch Module.
            Encode the input sequence by using attention mechanism.

        Args:
            hidden-states (:obj:`torch.Tensor` of shape ``(batch, dim_model, seq_enc)``): Input of encoder, might be the embedding of a batch of sequences. 
            attention_mask (:obj:`torch.Tensor` of shape ``(batch, seq_enc, seq_enc)``): Avoid invalid areas to participate in the calculation 
            position_bias(:obj:`torch.Tensor` of shape ``(num_heads, seq_enc, seq_enc)``) Provides position information to attention mechanism.  

        Return:
            out (:obj:`torch.Tensor` of shape ``(batch, dim_model, seq_enc)``): The encoder output. 

        """
        # (batch, dim_model, seq_enc)
        hidden_states = self.layers(hidden_states, attention_mask, position_bias, None, None, None)
        # (batch, dim_model, seq_enc)
        hidden_states = self.output_layernorm(hidden_states)
        return hidden_states


class Decoder(torch.nn.Module):
    def __init__(self, 
            num_layers : int,
            dim_model : int, 
            dim_ff : int,
            num_heads : int,
            dim_head : int,
            dtype : torch.dtype = torch.half,
            int8 : bool = False, 
            norm_init_var = 1.0,
            norm_bias = False,
            norm_eps : float = 1e-5, 
            att_init_mean = 0.0, 
            att_init_std = 0.02,
            att_bias = False,
            att_mask_value = float("-inf"),
            ffn_init_mean = 0.0, 
            ffn_init_std = 0.02,
            ffn_bias = False,
            ffn_activate_fn = "gated_gelu",
            pos_bias_type = "none",
            length_scale : bool = False,
            attn_scale : bool = False,
            dropout_p = 0,
            parallel_ffn = False,
        ):

        super().__init__()
        
        self.num_layers = num_layers

        self.layers = bmt.TransformerBlockList([
            bmt.CheckpointBlock(
                TransformerBlock(
                    dim_model = dim_model, 
                    dim_ff = dim_ff,
                    num_heads = num_heads,
                    dim_head = dim_head,
                    is_decoder = True,
                    dtype = dtype, 
                    int8 = int8,
                    norm_init_var = norm_init_var,
                    norm_bias = norm_bias,
                    norm_eps = norm_eps, 
                    att_init_mean = att_init_mean, 
                    att_init_std = att_init_std,
                    att_bias = att_bias,
                    att_mask_value = att_mask_value,
                    ffn_init_mean = ffn_init_mean, 
                    ffn_init_std = ffn_init_std,
                    ffn_bias = ffn_bias,
                    ffn_activate_fn = ffn_activate_fn,
                    pos_bias_type = pos_bias_type,
                    length_scale = length_scale,
                    attn_scale = attn_scale,
                    dropout_p = dropout_p,
                    parallel_ffn = parallel_ffn,
                )
            )
            for _ in range(num_layers)
        ])

        self.output_layernorm = LayerNorm(
                    dim_norm = dim_model, 
                    bias = norm_bias, 
                    dtype = dtype,
                    eps = norm_eps, 
                    init_var = norm_init_var)

    def forward(self, hidden_states : torch.Tensor,     # (batch, dim_model, seq_dec)
                      attention_mask : torch.Tensor,    # (batch, seq_dec, seq_dec)
                      position_bias : torch.Tensor,     # (num_heads, seq_dec, seq_dec)
                      cross_hidden_states = None,       # (batch, dim_model, seq_enc)
                      cross_attention_mask = None,      # (batch, seq_enc, seq_dec)
                      cross_position_bias = None,
                      ):
        """ This class is a PyTorch torch.nn.Module subclass.You can use it as a regular PyTorch Module.
            Decoder part of transformer.

        Args:
            hidden-states (:obj:`torch.Tensor` of shape ``(batch, dim_model, seq_dec)``): Input of decoder, Can be the embedding of a batch of sequences. 
            attention_mask (:obj:`torch.Tensor` of shape ``(batch, seq_dec, seq_dec)``): Avoid invalid areas to participate in the calculation. 
            position_bias(:obj:`torch.Tensor` of shape ``(num_heads, seq_dec, seq_dec)``) Provides position information to attention mechanism. 
            cross_hidden-states (:obj:`torch.Tensor` of shape ``(batch, dim_model, seq_dec)``): Input of decoder, Can be the output of encoder. 
            cross_attention_mask (:obj:`torch.Tensor` of shape ``(batch, seq_enc, seq_dec)``): Avoid invalid areas to participate in the calculation when the output of encoder participates in the calculation. 
            cross_position_bias(:obj:`torch.Tensor` of shape ``(num_heads, seq_enc, seq_dec)``) Provides position information to attention mechanism when the output of encoder participates in the calculation.  

        Return:
            out (:obj:`torch.Tensor` of shape ``(batch, dim_model, seq_dec)``): The decoder output. 

        """
        # (batch, dim_model, seq_dec)
        hidden_states = self.layers(hidden_states, attention_mask, position_bias,
                                    cross_hidden_states, cross_attention_mask, cross_position_bias)
        # (batch, dim_model, seq_dec)
        hidden_states = self.output_layernorm(hidden_states)
        return hidden_states
