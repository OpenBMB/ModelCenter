import torch
import bmtrain as bmp
import cpm_kernels.torch as ct
from cpm_kernels.torch.embedding import OpEmbedding
import math


class Embedding(bmp.DistributedModule):
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
        self.weight = bmp.DistributedParameter(
            torch.empty(vocab_size, embedding_size, dtype=dtype),
            init_method = bmp.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std)
        )
        self.length_scale = length_scale
        self.int8 = int8

    def forward(self, ids : torch.Tensor):
        """
        Args:
            ids : (batch_size, seq_len)                         int32
        Returns:
            embedding : (batch_size, embedding_size, seq_len)   fp16
        """
        embeds = OpEmbedding.apply(ids, self.weight)
        if self.length_scale:
            embeds = embeds / math.sqrt(self.dim_model)
        return embeds
    
    def projection(self, x : torch.Tensor):
        """
        Args:
            hidden : (batch_size, dim_model, seq_len)           int32
        Returns:
            logits : (batch, seq_len, vocab_output_size)        fp16
        """
        if self.length_scale:
            x = x / math.sqrt(self.dim_model)
        logits = ct.bmm(self.weight.unsqueeze(0), False, x, False, int8=self.int8) 
        logits = ct.transpose(logits)   # eqauls to .transpose(1, 2)
        return logits
