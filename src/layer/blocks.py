import torch
import cpm_kernels.torch as ct

from .attention import Attention
from .layernorm import LayerNorm
from .feedforward import FeedForward
import bmtrain as bmt
from typing import *


class SelfAttentionBlock(torch.nn.Module):

    def __init__(self, 
                 dim_model : int, 
                 num_heads : int, 
                 dim_head : int, 
                 dtype = torch.half,
                 int8 = False, 
                 norm_init_var = 1.0,
                 norm_bias = False,
                 norm_eps = 1e-5, 
                 att_init_mean = 0.0, 
                 att_init_std = 0.02,
                 att_bias = False,
                 att_mask_value = float("-inf"),
                 pos_bias_type = "none",
                 post_layer_norm = False,
                 length_scale : bool = False,
                 attn_scale : bool = False,
                 dropout_p = None):

        super().__init__()

        self.layernorm_before_attention = LayerNorm(
            dim_norm = dim_model, 
            bias = norm_bias, 
            dtype = dtype,
            eps = norm_eps, 
            init_var = norm_init_var)

        self.self_attention = Attention(
            dim_in = dim_model, 
            num_heads = num_heads, 
            dim_head = dim_head,
            dim_out = dim_model, 
            dtype = dtype,
            int8 = int8, 
            init_mean = att_init_mean,
            init_std = att_init_std,
            bias = att_bias,
            mask_value = att_mask_value,
            pos_bias_type = pos_bias_type,
            length_scale = length_scale,
            attn_scale = attn_scale,
            dropout_p = dropout_p)

        if dropout_p is None:
            self.dropout = None
        else:
            self.dropout = torch.nn.Dropout(dropout_p)

        self.post_layer_norm = post_layer_norm

    def forward(self,
                hidden_states : torch.Tensor,       # (batch, dim_model, seq_self)
                attention_mask : torch.Tensor,      # (batch, seq_self, seq_self)
                position_bias : Optional[torch.Tensor] = None,       # (num_heads, seq_self, seq_self)
            ):
              
        x = self.layernorm_before_attention(hidden_states)
        if self.post_layer_norm:
            hidden_states = x
        x = self.self_attention(x, x, attention_mask, position_bias)
        if self.dropout is not None:
            x = self.dropout(x)
        hidden_states = ct.element_add(hidden_states, x)
        return hidden_states


class CrossAttentionBlock(torch.nn.Module):

    def __init__(self, 
                 dim_model : int, 
                 num_heads : int, 
                 dim_head : int, 
                 dtype = torch.half,
                 int8 = False, 
                 norm_init_var = 1.0,
                 norm_bias = False,
                 norm_eps = 1e-5, 
                 att_init_mean = 0.0, 
                 att_init_std = 0.02,
                 att_bias = False,
                 att_mask_value = float("-inf"),
                 pos_bias_type = "none",
                 post_layer_norm = False,
                 length_scale : bool = False,
                 attn_scale : bool = False,
                 dropout_p = None):

        super().__init__()

        self.layernorm_before_attention = LayerNorm(
            dim_norm = dim_model, 
            bias = norm_bias, 
            dtype = dtype,
            eps = norm_eps, 
            init_var = norm_init_var)

        self.self_attention = Attention(
            dim_in = dim_model, 
            num_heads = num_heads, 
            dim_head = dim_head,
            dim_out = dim_model, 
            dtype = dtype,
            int8 = int8, 
            init_mean = att_init_mean,
            init_std = att_init_std,
            bias = att_bias,
            mask_value = att_mask_value,
            pos_bias_type = pos_bias_type,
            length_scale = length_scale,
            attn_scale = attn_scale,
            dropout_p = dropout_p)

        if dropout_p is None:
            self.dropout = None
        else:
            self.dropout = torch.nn.Dropout(dropout_p)

        self.post_layer_norm = post_layer_norm

    def forward(self,
                hidden_states : torch.Tensor,       # (batch, dim_model, seq_self)
                key_value_states: torch.Tensor,     # (batch, dim_model, seq_cross)
                attention_mask : torch.Tensor,      # (batch, seq_cross, seq_self)
                position_bias : Optional[torch.Tensor] = None,
            ):

        x = self.layernorm_before_attention(hidden_states)
        if self.post_layer_norm:
            hidden_states = x
        x = self.self_attention(x, key_value_states, attention_mask, position_bias)
        if self.dropout is not None:
            x = self.dropout(x)
        hidden_states = ct.element_add(hidden_states, x)
        return hidden_states


class FFNBlock(torch.nn.Module):
    def __init__(self, 
                 dim_model : int, 
                 dim_ff : int,
                 dtype = torch.half, 
                 int8 = False,
                 norm_init_var = 1.0,
                 norm_bias = False,
                 norm_eps = 1e-5, 
                 ffn_init_mean = 0.0, 
                 ffn_init_std = 0.02,
                 ffn_bias = False,
                 ffn_activate_fn = "gated_gelu",
                 post_layer_norm = False,
                 length_scale : bool = False,
                 dropout_p = None):

        super().__init__()

        self.layernorm_before_ffn = LayerNorm(
            dim_norm = dim_model, 
            bias = norm_bias, 
            dtype = dtype,
            eps = norm_eps, 
            init_var = norm_init_var,
        )

        self.ffn = FeedForward(
            dim_in = dim_model, 
            dim_ff = dim_ff, 
            dim_out = dim_model, 
            dtype = dtype, 
            int8 = int8,
            init_mean = ffn_init_mean, 
            init_std = ffn_init_std,
            bias = ffn_bias,
            activate_fn = ffn_activate_fn,
            length_scale = length_scale,
        )

        if dropout_p is None:
            self.dropout = None
        else:
            self.dropout = torch.nn.Dropout(dropout_p)

        self.post_layer_norm = post_layer_norm

    def forward(self,
                hidden_states : torch.Tensor,   # (batch, dim_model, seq_self)
               ):

        x = self.layernorm_before_ffn(hidden_states)
        if self.post_layer_norm:
            hidden_states = x
        x = self.ffn(x)
        if self.dropout is not None:
            x = self.dropout(x)
        hidden_states = ct.element_add(hidden_states, x)
        return hidden_states


class TransformerBlock(torch.nn.Module):

    def __init__(self, 
                 dim_model : int, 
                 dim_ff : int,
                 num_heads : int,
                 dim_head : int,
                 is_decoder = False,
                 dtype = torch.half, 
                 int8 = False,
                 norm_init_var = 1.0,
                 norm_bias = False,
                 norm_eps = 1e-5, 
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
                 parallel_ffn = False,
                 length_scale : bool = False,
                 attn_scale : bool = False,
                 dropout_p = None):

        super().__init__()

        self.is_decoder = is_decoder

        self.self_att = SelfAttentionBlock(
                 dim_model = dim_model, 
                 num_heads = num_heads, 
                 dim_head = dim_head, 
                 dtype = dtype,
                 int8 = int8, 
                 norm_eps = norm_eps, 
                 norm_init_var = norm_init_var,
                 norm_bias = norm_bias,
                 att_init_mean = att_init_mean, 
                 att_init_std = att_init_std,
                 att_bias = att_bias,
                 att_mask_value = att_mask_value,
                 pos_bias_type = pos_bias_type,
                 post_layer_norm = post_layer_norm,
                 length_scale = length_scale,
                 attn_scale = attn_scale,
                 dropout_p = dropout_p)

        if is_decoder:
            self.cross_att = CrossAttentionBlock(
                 dim_model = dim_model, 
                 num_heads = num_heads, 
                 dim_head = dim_head, 
                 dtype = dtype,
                 int8 = int8, 
                 norm_eps = norm_eps, 
                 norm_init_var = norm_init_var,
                 norm_bias = norm_bias,
                 att_init_mean = att_init_mean, 
                 att_init_std = att_init_std,
                 att_bias = att_bias,
                 att_mask_value = att_mask_value,
                 pos_bias_type = pos_bias_type,
                 length_scale = length_scale,
                 attn_scale = attn_scale,
                 dropout_p = dropout_p)
        else:
            self.cross_att = None

        self.ffn = FFNBlock(
                 dim_model = dim_model, 
                 dim_ff = dim_ff,
                 dtype = dtype, 
                 int8 = int8,
                 norm_eps = norm_eps, 
                 norm_init_var = norm_init_var,
                 norm_bias = norm_bias,
                 ffn_init_mean = ffn_init_mean, 
                 ffn_init_std = ffn_init_std,
                 ffn_bias = ffn_bias,
                 ffn_activate_fn = ffn_activate_fn,
                 length_scale = length_scale,
                 dropout_p = dropout_p,
                 post_layer_norm = post_layer_norm)

        self.parallel_ffn = parallel_ffn

    def forward(self,
                self_hidden_states : torch.Tensor,       # (batch, dim_model, seq_self)
                self_attention_mask : torch.Tensor,      # (batch, seq_self, seq_self)
                self_position_bias : Optional[torch.Tensor] = None,       # (num_heads, seq_self, seq_self)
                cross_hidden_states = None,              # (batch, dim_model, seq_cross)
                cross_attention_mask = None,             # (batch, seq_cross, seq_self)
                cross_position_bias = None,
            ):

        # (batch, dim_model, seq_self)
        hidden_states = self.self_att(self_hidden_states,
                                      attention_mask = self_attention_mask,
                                      position_bias = self_position_bias)

        # (batch, dim_model, seq_self)
        if self.is_decoder and self.cross_att is not None:
            hidden_states = self.cross_att(hidden_states = hidden_states,
                                           key_value_states = cross_hidden_states,
                                           attention_mask = cross_attention_mask,
                                           position_bias = cross_position_bias)

        # (batch, dim_model, seq_self)
        if self.parallel_ffn:
            hidden_states_2 = self.ffn(self_hidden_states)
            hidden_states = ct.element_add(ct.element_add(hidden_states, -self_hidden_states), hidden_states_2) # hidden states calculate twice in residual, minus one
        else:
            hidden_states = self.ffn(hidden_states)
        return hidden_states

