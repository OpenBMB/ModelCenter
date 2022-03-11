import torch
import bmtrain as bmt
from cpm_kernels.torch.position_embedding import OpPositionEmbedding


class RelativePositionEmbedding(bmt.DistributedModule):

    def __init__(self, num_heads, 
                       num_buckets = 32, 
                       max_distance = 128, 
                       bidirectional = False, 
                       dtype = torch.half,
                       init_mean = 0.0,
                       init_std = 1,
                    ):

        super().__init__()

        self.relative_attention_bias = bmt.DistributedParameter(
            torch.randn(num_heads, num_buckets, dtype = dtype), 
            init_method = bmt.ParameterInitializer(torch.nn.init.normal_, mean = init_mean, std = init_std)
        )
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional

    def forward(self, key_len, query_len):
        """
        Args:
            key_len: int
            query_len : int
        Returns:
            out : (num_heads, key_len, query_len)   fp16
        """
        return OpPositionEmbedding.apply(
            query_len, 
            key_len, 
            self.num_buckets, 
            self.max_distance, 
            self.num_heads, 
            self.relative_attention_bias, 
            self.bidirectional
        )


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, rotary_dim):
        super().__init__()
        self.rotary_dim = rotary_dim

    def fixed_pos_embedding(self, x, seq_dim=1, seq_len=None):
        dim = x.shape[-2]
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
        sinusoid_inp = torch.einsum("j , i -> i j", torch.arange(seq_len), inv_freq).to(x.device).half()
        return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)
    
    def rotate_every_two(self, x):
        x1 = x[:, ::2, :]
        x2 = x[:, 1::2, :]
        x = torch.stack((-x2, x1), axis=-2)
        return x.flatten(-3, -2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')
    
    def apply_rotary_pos_emb(self, x, sincos, offset=0):
        sin, cos = map(lambda t: t[None, :, offset : x.shape[-1] + offset].repeat_interleave(2, 1), sincos)
        # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
        return (x * cos) + (self.rotate_every_two(x) * sin)

    def forward(self, h_q, h_k):
        k_rot = h_k[:, : self.rotary_dim, :]
        k_pass = h_k[:, self.rotary_dim :, :]

        q_rot = h_q[:, : self.rotary_dim, :]
        q_pass = h_q[:, self.rotary_dim :, :]

        seq_len = h_k.shape[-1]
        sincos = self.fixed_pos_embedding(k_rot, -1, seq_len=seq_len)
        k_rot = self.apply_rotary_pos_emb(k_rot, sincos, offset=0)
        q_rot = self.apply_rotary_pos_emb(q_rot, sincos, offset=0)

        h_k = torch.cat([k_rot, k_pass], dim=-2)
        h_q = torch.cat([q_rot, q_pass], dim=-2)
        return h_q, h_k