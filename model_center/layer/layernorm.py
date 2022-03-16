import torch
import bmtrain as bmt
import torch.nn.functional as F

@torch.jit.script
def rms_layernorm(hidden : torch.Tensor, weight : torch.Tensor, eps :float):
    old_dtype = hidden.dtype
    variance = hidden.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    hidden = (hidden * torch.rsqrt(variance + eps)).to(old_dtype)
    return hidden * weight


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
        """
        Args:
            x: (batch_size, seq_len, dim_norm)
        
        Returns:
            out : (batch_size, seq_len, dim_norm)
        """
        assert x.size(-1) == self.dim_norm
        
        if self.bias is not None:
            return F.layer_norm(x, (self.dim_norm,), self.weight, self.bias, self.eps)
        else:
            return rms_layernorm(x, self.weight, self.eps)
