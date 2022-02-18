import torch
import bmtrain as bmp
from cpm_kernels.torch.layernorm import OpLayerNormMean, OpLayerNormNoMean


class LayerNorm(bmp.DistributedModule):
    def __init__(self, dim_norm : int, 
                       dtype=torch.half, 
                       bias=True, 
                       eps : float = 1e-5,
                       init_var = 1.0
                       ):

        super().__init__()

        self.eps = eps
        self.dim_norm = dim_norm
        self.weight = bmp.DistributedParameter(
            torch.ones(dim_norm, dtype=dtype) * init_var)
        self.bias = bmp.DistributedParameter(
            torch.zeros(dim_norm, dtype=dtype)) if bias else None
    
    def forward(self, x : torch.Tensor):
        """
        Args:
            x: (batch_size, dim_norm, seq_len)       fp16
        
        Returns:
            out : (batch_size, dim_norm, seq_len)    fp16
        """
        assert x.size(1) == self.dim_norm
        
        if self.bias is not None:
            return OpLayerNormMean.apply(x, self.eps, self.weight, self.bias)
        else:
            return OpLayerNormNoMean.apply(x, self.eps, self.weight)
