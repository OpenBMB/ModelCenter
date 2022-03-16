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

class Projection(bmt.DistributedModule):
    def __init__(self,
                 dim_out : int,
                 dim_in : int,
                 length_scale : bool = False,
                 dtype = torch.half,
                 int8 = False,
                 init_mean = 0.0,
                 init_std = 1,
                 bias = False,
                ):
        super().__init__()

        self.w = Linear(
            dim_in = dim_in,
            dim_out = dim_out,
            length_scale = length_scale,
            length_scale_before = True,
            dtype = dtype,
            int8 = int8,
            init_mean = init_mean,
            init_std = init_std,
            bias = bias,
        )

    def forward(self, x : torch.Tensor):
        """ This class inherits from DistributedModule.
            The projection class can be used to project input tensors in `dim_in`-dimensional space to output tensors in `dim_out`-dimensional vector space 
            The projection is achieved through a linear layer 

        Args:
            x (:obj:`torch.Tensor` of shape ``(batch_size, dim_in, seq_len)``): The tensor you want to perform the projection operation on.

        Return:
            logits (:obj:`torch.Tensor` of shape ``(batch_size, seq_len, dim_out)``): The output of the projection layer.

        """
        logits = self.w(x)
        logits = ct.transpose(logits)   # eqauls to .transpose(1, 2)
        return logits
