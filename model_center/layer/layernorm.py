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
from cpm_kernels.torch.layernorm import OpLayerNormMean, OpLayerNormNoMean


class LayerNorm(bmt.DistributedModule):
    def __init__(self, dim_norm : int, 
                       dtype=torch.half, 
                       bias=True, 
                       eps : float = 1e-5,
                       init_var = 1.0
                       ):

        super().__init__()

        self.eps = eps
        self.dim_norm = dim_norm
        self.weight = bmt.DistributedParameter(
            torch.ones(dim_norm, dtype=dtype) * init_var)
        self.bias = bmt.DistributedParameter(
            torch.zeros(dim_norm, dtype=dtype)) if bias else None
    
    def forward(self, x : torch.Tensor):
        """ This model inherits from bmt.DistributedModule. 
            Used to normalize each training sample to the same distribution 

        Args:
            x (:obj:`torch.Tensor` of shape ``(batch_size, dim_norm, seq_len)``): Input tensor that need to be normalized to be put in the further calculation.

        Return:
            out (:obj:`torch.Tensor` of shape ``(batch_size, dim_norm, seq_len)``): The layernorm output. 

        """
        assert x.size(1) == self.dim_norm
        
        if self.bias is not None:
            return OpLayerNormMean.apply(x, self.eps, self.weight, self.bias)
        else:
            return OpLayerNormNoMean.apply(x, self.eps, self.weight)
