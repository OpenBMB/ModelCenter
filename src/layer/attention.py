import math
from typing import Optional

import torch
import bmtrain as bmp
import cpm_kernels.torch as ct
from .linear import Linear


class Attention(bmp.DistributedModule):
    def __init__(self, dim_in : int, 
                       dim_head : int,
                       num_heads : int, 
                       dim_out = None,
                       dtype = torch.half,
                       int8 = True, 
                       init_mean = 0.0, 
                       init_std = 0.02,
                       bias = False,
                       mask_value = float("-inf"),
                       pos_bias_type = "none",
                       length_scale : bool = False,
                       attn_scale : bool = False,
                       dropout_p = None,
                       ):

        super().__init__()

        if dim_out is None:
            dim_out = dim_in

        self.project_q = Linear(
            dim_in = dim_in,
            dim_out = num_heads * dim_head,
            length_scale = length_scale,
            length_scale_before = False,
            dtype = dtype,
            int8 = int8,
            init_mean = init_mean,
            init_std = init_std,
            bias = bias,
        )

        self.project_k = Linear(
            dim_in = dim_in,
            dim_out = num_heads * dim_head,
            length_scale = length_scale,
            length_scale_before = False,
            dtype = dtype,
            int8 = int8,
            init_mean = init_mean,
            init_std = init_std,
            bias = bias,
        )

        self.project_v = Linear(
            dim_in = dim_in,
            dim_out = num_heads * dim_head,
            length_scale = length_scale,
            length_scale_before = False,
            dtype = dtype,
            int8 = int8,
            init_mean = init_mean,
            init_std = init_std,
            bias = bias,
        )

        self.attention_out = Linear(
            dim_in = num_heads * dim_head,
            dim_out = dim_out,
            length_scale = length_scale,
            length_scale_before = False,
            dtype = dtype,
            int8 = int8,
            init_mean = init_mean,
            init_std = init_std,
            bias = bias,
        )
    
        self.dim_in = dim_in
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dim_out = dim_out
        self.int8 = int8
        self.length_scale = length_scale
        self.attn_scale = attn_scale
        self.mask_value = mask_value

        if dropout_p is None:
            self.attention_dropout = None
        else:
            self.attention_dropout = torch.nn.Dropout(dropout_p)

        self.pos_bias_type = pos_bias_type

    def forward(self, 
            query : torch.Tensor,                   # (batch, dim_model, len_q)
            key_value : torch.Tensor,               # (batch, dim_model, len_k)
            mask : torch.Tensor,                    # (batch, len_k, len_q)
            position_bias : Optional[torch.Tensor] = None  # (num_heads, len_k, len_q) or (1, num_heads, len_k, len_q) 
        ):
        """
        Args:
            query : (batch, dim_model, len_q)           fp16
            key_value : (batch, dim_model, len_k)       fp16
            mask : (batch, len_k, len_q)                fp16
            position_bias : (num_heads, len_k, len_q)   fp16
        Returns:
            out : (batch, dim_model, len_q)             fp16
        """

        batch_size = query.size(0)
        len_q = query.size(2)
        len_k = key_value.size(2)

        # (1#batch, num_heads * dim_head, dim_model) @ (batch, dim_model, len_q) 
        # => (batch, num_heads * dim_head, len_q)
        h_q = self.project_q(query)
        h_k = self.project_k(key_value)
        h_v = self.project_v(key_value)

        # view (batch * num_heads, dim_head, length)
        h_q = h_q.view(batch_size * self.num_heads, self.dim_head, -1)
        h_k = h_k.view(batch_size * self.num_heads, self.dim_head, -1)
        h_v = h_v.view(batch_size * self.num_heads, self.dim_head, -1)

        if self.pos_bias_type == "rotary":
            h_q, h_k = position_bias(h_q, h_k)

        # (batch * num_heads, dim_head, len_k)^T @ (batch * num_heads, dim_head, len_q) 
        # => (batch * num_heads, len_k, len_q)
        score = ct.bmm(h_k, True, h_q, False, int8 = False) # use FP 16 here
        if self.attn_scale:
            score = score / math.sqrt(self.dim_head)

        # (batch, num_heads, len_k, len_q) 
        score = score.view(batch_size, self.num_heads, len_k, len_q)

        if self.pos_bias_type == "relative":
            # (batch, num_heads, len_k, len_q) + (1, num_heads, len_k, len_q) 
            if position_bias.dim() == 3:
                score = ct.batched_add(score, position_bias)
            else:
                score = ct.element_add(score, position_bias)

        # (batch, num_heads, len_k * len_q)
        masked_score = ct.mask(
            score.view(batch_size, self.num_heads, -1),
            mask.view(batch_size, -1),
            self.mask_value,
        )

        # (batch * num_heads, len_k, len_q)
        masked_score = masked_score.view(batch_size * self.num_heads, len_k, len_q)

        # (batch * num_heads, len_k, len_q)
        masked_score = ct.softmax(masked_score) # softmax along len_k

        if self.attention_dropout is not None:
            masked_score = self.attention_dropout(masked_score)

        # (batch * num_heads, dim_head, len_k) @ (batch * num_heads, len_k, len_q) = (batch * num_heads, dim_head, len_q)
        attention_result = ct.bmm(h_v, False, masked_score, False, int8=False)  # use FP 16 here

        attention_result = attention_result.view(batch_size, self.num_heads * self.dim_head, len_q)

        # (1#batch, dim_model, num_heads * dim_head) @ (batch, num_heads * dim_head, len_q) = (batch, dim_model, len_q)
        attention_out = self.attention_out(attention_result)

        return attention_out
