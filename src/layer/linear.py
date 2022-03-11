import torch
import bmtrain as bmp
import cpm_kernels.torch as ct
import math
import torch.nn.functional as F

class Linear(bmp.DistributedModule):
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
        self.weight = bmp.DistributedParameter(
            torch.empty((dim_out, dim_in), dtype=dtype),
            init_method=bmp.ParameterInitializer(torch.nn.init.normal_, mean=init_mean, std=init_std)
        )
        self.bias = bmp.DistributedParameter(
            torch.empty((dim_out,), dtype=dtype),
            init_method=bmp.ParameterInitializer(torch.nn.init.zeros_)
        ) if bias else None
        self.length_scale = length_scale
        self.length_scale_before = length_scale_before
        self.int8 = int8

    def forward(self, x : torch.Tensor):
        """
        Args:
            hidden : (batch_size, dim_in, seq_len)           int32
        Returns:
            logits : (batch, dim_out, seq_len)        fp16
        """
        if self.length_scale and self.length_scale_before:
            x = x / math.sqrt(self.dim_in)
        # x = ct.bmm(self.weight.unsqueeze(0), False, x, False, int8=self.int8) 
        x = ct.transpose(F.linear(ct.transpose(x), self.weight))
        if self.length_scale and not self.length_scale_before:
            x = x / math.sqrt(self.dim_in)
        if self.bias is not None:
            x = ct.ln_add(x, self.bias)
        return x
