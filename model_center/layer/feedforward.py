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
import bmtrain as bmt
import cpm_kernels.torch as ct
from .linear import Linear
import math

class DenseGatedACT(bmt.DistributedModule):

    def __init__(self,
                 dim_in : int,
                 dim_ff : int,
                 activate_fn : str = "gelu",
                 dtype = torch.half,
                 int8 = False,
                 init_mean = 0.0,
                 init_std = 0.02,
                 bias = False,
                 length_scale : bool = False,
        ):
        super().__init__()

        self.w_0 = Linear(
            dim_in = dim_in,
            dim_out = dim_ff,
            length_scale = length_scale,
            length_scale_before = False,
            dtype = dtype,
            int8 = int8,
            init_mean = init_mean,
            init_std = init_std,
            bias = bias,
        )

        self.w_1 = Linear(
            dim_in = dim_in,
            dim_out = dim_ff,
            length_scale = length_scale,
            length_scale_before = False,
            dtype = dtype,
            int8 = int8,
            init_mean = init_mean,
            init_std = init_std,
            bias = bias,
        )

        if activate_fn == "relu":
            self.act = torch.nn.ReLU()
        elif activate_fn == "gelu":
            self.act = ct.gelu
        else:
            raise ValueError("Unsupported activation function: %s" % (activate_fn))
    
    def forward(self, x):
        # (batch, dim_ff, dim_in) @ (batch, dim_in, seq_len) 
        # => (batch, dim_ff, seq_len)
        """ This model inherits from BaseModel. 
            Transform an input tensor from one feature space to another via a nonlinear operation
        
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, dim_in, seq_length)``): Tensor that will be subject to nonlinear operations.

        Return:
            out (:obj:`torch.Tensor` of shape ``(batch, dim_ff, seq_len)``) 

        """
        gelu_score = self.act( self.w_0(x) )
        hidden_out = self.w_1(x)

        x = ct.element_mul(gelu_score, hidden_out)

        return x


class DenseACT(bmt.DistributedModule):

    def __init__(self,
                 dim_in : int,
                 dim_ff : int,
                 activate_fn : str = "gelu",
                 dtype = torch.half,
                 int8 = False,
                 init_mean = 0.0,
                 init_std = 0.02,
                 bias = False,
                 length_scale : bool = False,
        ):
        super().__init__()

        self.w = Linear(
            dim_in = dim_in,
            dim_out = dim_ff,
            length_scale = length_scale,
            length_scale_before = False,
            dtype = dtype,
            int8 = int8,
            init_mean = init_mean,
            init_std = init_std,
            bias = bias,
        )
        
        if activate_fn == "relu":
            self.act = torch.nn.ReLU()
        elif activate_fn == "gelu":
            self.act = ct.gelu
        else:
            raise ValueError("Unsupported activation function: %s" % (activate_fn))

    def forward(self, x):
        # (batch, dim_ff, dim_in) @ (batch, dim_in, seq_len) 
        # => (batch, dim_ff, seq_len)
        """ This model inherits from BaseModel. 
            Transform an input tensor from one feature space to another via a nonlinear operation
        
        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, dim_in, seq_length)``): Tensor that will be subject to nonlinear operations.

        Return:
            out (:obj:`torch.Tensor` of shape ``(batch, dim_ff, seq_len)``) 

        """
        x = self.w(x)
        x = self.act(x)
        
        return x

class FeedForward(bmt.DistributedModule):

    def __init__(self,
                 dim_in : int, 
                 dim_ff : int,
                 dim_out = None,
                 dtype = torch.half, 
                 int8 = False,
                 init_mean = 0.0, 
                 init_std = 0.02,
                 bias = False,
                 activate_fn = "gated_gelu",
                 length_scale : bool = False,
                 dropout_p = 0,
        ):

        super().__init__()

        if activate_fn.startswith("gated_"):
            self.w_in = DenseGatedACT(
                dim_in = dim_in,
                dim_ff = dim_ff,
                activate_fn = activate_fn[6:],
                dtype = dtype,
                int8 = int8,
                init_mean = init_mean,
                init_std = init_std,
                bias = bias,
                length_scale = length_scale,
            )
        else:
            self.w_in = DenseACT(
                dim_in = dim_in,
                dim_ff = dim_ff,
                activate_fn = activate_fn,
                dtype = dtype,
                int8 = int8,
                init_mean = init_mean,
                init_std = init_std,
                bias = bias,
                length_scale = length_scale,
            )

        if dropout_p:
            self.dropout = torch.nn.Dropout(dropout_p)
        else:
            self.dropout = None

        if dim_out is None:
            dim_out = dim_in

        self.dim_ff = dim_ff
        self.dim_out = dim_out

        self.w_out = Linear(
            dim_in = dim_ff,
            dim_out = dim_out,
            length_scale = length_scale,
            length_scale_before = True,
            dtype = dtype,
            int8 = int8,
            init_mean = init_mean,
            init_std = init_std,
            bias = bias,
        )

        self.int8 = int8
        self.length_scale = length_scale

    def forward(self, x):
        """ 
            This model inherits from bmt.DistributedModule.
            In order to add nonlinear operation on the tensor.

        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, dim_in, seq_len)``): Tensor that will be sent to feed forward layer.

        Return:
            out (:obj:`torch.Tensor` of shape ``(batch, dim_out, seq_len)``): The feed-forward output.

        """
        x = self.w_in(x)

        if self.dropout is not None:
            x = self.dropout(x)

        # (batch, dim_out, dim_ff) @ (batch, dim_ff, seq_len) 
        # => (batch, dim_out, seq_len)
        x = self.w_out(x)

        return x
