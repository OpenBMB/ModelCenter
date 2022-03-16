import math
import torch
import bmtrain as bmt
import torch.nn.functional as F

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
            torch.empty(num_buckets, num_heads, dtype = dtype), 
            init_method = bmt.ParameterInitializer(torch.nn.init.normal_, mean = init_mean, std = init_std)
        )
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.bidirectional = bidirectional

    def forward(self, query_len, key_len):
        """
        Args:
            query_len : int
            key_len: int
        Returns:
            out : (num_heads, query_len, key_len)
        """
        part_buckets = self.num_buckets // (2 if self.bidirectional else 1)
        exact_buckets = part_buckets // 2
        log_buckets = part_buckets - exact_buckets

        pos_q = torch.arange(query_len, dtype=torch.long, device="cuda")
        pos_k = torch.arange(key_len, dtype=torch.long, device="cuda")
        relative_position = pos_q[:, None] - pos_k[None, :] # (query_len, key_len)

        neg_pos = relative_position < 0
        relative_position = relative_position.abs()

        small_pos = relative_position < exact_buckets

        log_pos = (torch.clamp(
            torch.log(relative_position.float() / exact_buckets) / math.log(self.max_distance / exact_buckets),
            0,
            0.9999
        ) * log_buckets).long() + exact_buckets

        buckets = torch.where(small_pos, relative_position, log_pos)
        if self.bidirectional:
            buckets = torch.where(
                neg_pos,
                buckets + part_buckets,
                buckets
            )
        else:
            buckets = torch.masked_fill(
                buckets,
                neg_pos,
                0,
            )
        return F.embedding(buckets, self.relative_attention_bias, padding_idx = -1).permute(2, 0, 1).contiguous()


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, rotary_dim):
        super().__init__()
        self.rotary_dim = rotary_dim

    def fixed_pos_embedding(self, x, seq_dim=1, seq_len=None, dtype = torch.float):
        dim = x.shape[-1]
        if seq_len is None:
            seq_len = x.shape[seq_dim]
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
        sinusoid_inp = torch.einsum("i , j -> i j", torch.arange(seq_len), inv_freq).to(x.device).to(dtype)
        return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)

    def rotate_every_two(self, x):
        x1 = x[:, :, :, ::2]
        x2 = x[:, :, :, 1::2]
        x = torch.stack((-x2, x1), axis=-1)
        return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')


    def apply_rotary_pos_emb(self, x, sincos, offset=0):
        sin, cos = map(lambda t: t[None, offset : x.shape[1] + offset, None, :].repeat_interleave(2, 3), sincos)
        # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
        return (x * cos) + (self.rotate_every_two(x) * sin)

    def forward(self, h_q, h_k):
        """
        Args:
            h_q : (batch_size, len_q, num_heads, dim_head)
            h_k : (batch_size, len_k, num_heads, dim_head)
        Returns:
            h_q : (batch_size, len_q, num_heads, dim_head)
            h_k : (batch_size, len_k, num_heads, dim_head)
        """
        k_rot = h_k[:, :, :, : self.rotary_dim]
        k_pass = h_k[:, :, :, self.rotary_dim :]

        q_rot = h_q[:, :, :, : self.rotary_dim]
        q_pass = h_q[:, :, :, self.rotary_dim :]

        seq_len = h_k.shape[1]
        sincos = self.fixed_pos_embedding(k_rot, 1, seq_len=seq_len, dtype=h_k.dtype)
        k_rot = self.apply_rotary_pos_emb(k_rot, sincos, offset=0)
        q_rot = self.apply_rotary_pos_emb(q_rot, sincos, offset=0)

        h_k = torch.cat([k_rot, k_pass], dim=-1)
        h_q = torch.cat([q_rot, q_pass], dim=-1)
        return h_k, h_q
