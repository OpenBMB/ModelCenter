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
        """
        Args:
            ids : (batch, seq_len)                         long
        Returns:
            embedding : (batch, seq_len, embedding_size)   
        """
        
        embeds = F.embedding(ids, self.weight)
        if self.length_scale:
            embeds = embeds / math.sqrt(self.dim_model)
        return embeds
    
    def projection(self, x : torch.Tensor):
        """
        Args:
            hidden : (batch, seq_len, dim_model)           int32
        Returns:
            logits : (batch, seq_len, vocab_output_size)        fp16
        """
        if self.length_scale:
            x = x / math.sqrt(self.dim_model)
        logits = F.linear(x, self.weight)
        return logits
