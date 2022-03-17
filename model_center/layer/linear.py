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
import math
import torch.nn.functional as F

class Linear(bmt.DistributedModule):
    def __init__(self,
                 dim_in : int,
                 dim_out : int,
                 length_scale : bool = False,
                 length_scale_before : bool = False,
                 dtype = torch.half,
                 int8 : bool = False,
                 init_mean : float = 0.0,
                 init_std : float = 1,
                 bias : bool = False,
                ):
        super().__init__()
        self.dim_in = dim_in
        self.weight = bmt.DistributedParameter(
            torch.empty((dim_out, dim_in), dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std)
        )
        self.bias = bmt.DistributedParameter(
            torch.empty((dim_out,), dtype=dtype),
            init_method=bmt.ParameterInitializer(torch.nn.init.zeros_)
        ) if bias else None
        self.length_scale = length_scale
        self.length_scale_before = length_scale_before
        self.int8 = int8

    def forward(self, x : torch.Tensor):
        """ This class inherits from bmt.DistributedModule.
            This is a fully connected layer that can be used to change the dimension, and can be used to get logits.

        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_in)``): Input of linear layer

        Return:
            logits (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_out)``): The linear layer output.

        """
        if self.length_scale and self.length_scale_before:
            x = x / math.sqrt(self.dim_in)
        x = F.linear(x, self.weight)
        if self.length_scale and not self.length_scale_before:
            x = x / math.sqrt(self.dim_in)
        if self.bias is not None:
            x = x + self.bias
        return x
