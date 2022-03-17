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

class Embedding(bmt.DistributedModule):
    def __init__(self,
                 vocab_size : int,
                 embedding_size : int,
                 length_scale : bool = False,
                 dtype = torch.half,
                 int8 = False,
                 init_mean = 0.0,
                 init_std = 1,
                ):
        super().__init__()
        self.dim_model = embedding_size
        self.weight = bmt.DistributedParameter(
            torch.empty(vocab_size, embedding_size, dtype=dtype),
            init_method = bmt.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std)
        )
        self.length_scale = length_scale
        self.int8 = int8

    def forward(self, ids : torch.Tensor):
        """ This class inherits from bmt.DistributedModule. 
            You can embed a sequence of indices through this layer.

        Args:
            ids (:obj:`torch.Tensor` of shape ``(batch_size, seq_len)``): Indices of input sequence tokens. It will be embedded by model's internal embedding lookup matrix.

        Return:
            out (:obj:`torch.Tensor` of shape ``(batch_size, seq_len, embedding_size)``): The embedding output.
        """
        
        embeds = F.embedding(ids, self.weight)
        if self.length_scale:
            embeds = embeds / math.sqrt(self.dim_model)
        return embeds
    
    def projection(self, x : torch.Tensor):
        """
        Projection based on embedding's weight. For example, embedding map vocab_size to embed_size, than projection map embed_size back to vocab_size.

        Args:
            x (:obj:`torch.Tensor` of shape ``(batch, seq_len, dim_model)``): Input of projection
        Returns:
            out (:obj:`torch.Tensor` of shape ``(batch, seq_len, vocab_output_size)``): The projection output.
        """
        if self.length_scale:
            x = x / math.sqrt(self.dim_model)
        logits = F.linear(x, self.weight)
        return logits
