import torch
from torch.nn.modules import dropout
import bmpretrain as bmp
import cpm_kernels.torch as ct
from .linear import Linear
import math

class DenseGatedACT(bmp.DistributedModule):

    def __init__(self,
                 dim_in : int,
                 dim_ff : int,
                 activate_fn : str = "gelu",
                 dtype = torch.half,
                 int8 = True,
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

        gelu_score = self.act( self.w_0(x) )
        hidden_out = self.w_1(x)

        x = ct.element_mul(gelu_score, hidden_out)

        return x


class DenseACT(bmp.DistributedModule):

    def __init__(self,
                 dim_in : int,
                 dim_ff : int,
                 activate_fn : str = "gelu",
                 dtype = torch.half,
                 int8 = True,
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

        x = self.w(x)
        x = self.act(x)
        
        return x

class FeedForward(bmp.DistributedModule):

    def __init__(self,
                 dim_in : int, 
                 dim_ff : int,
                 dim_out = None,
                 dtype = torch.half, 
                 int8 = True,
                 init_mean = 0.0, 
                 init_std = 0.02,
                 bias = False,
                 activate_fn = "gated_gelu",
                 length_scale : bool = False,
                 dropout_p = None,
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

        if dropout_p is None:
            self.dropout = None
        else:
            self.dropout = torch.nn.Dropout(dropout_p)

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
        Args:
            x : (batch, dim_in, seq_len)       fp16
        Returns:
            out : (batch, dim_out, seq_len)     fp16
        """

        x = self.w_in(x)

        if self.dropout is not None:
            x = self.dropout(x)

        # (batch, dim_out, dim_ff) @ (batch, dim_ff, seq_len) 
        # => (batch, dim_out, seq_len)
        x = self.w_out(x)

        return x
