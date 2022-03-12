import torch
from .config import Config

class GPT2Config(Config):

    def __init__(self, vocab_size=50258,
                       dim_model=768,
                       num_heads=12,
                       dim_head=64,
                       dim_ff=3072,
                       num_layers=12,
                       dropout_p=0.1,
                       emb_init_mean = 0.0,
                       emb_init_std = 1,
                       pos_bias_type = "none",
                       position_bias_max_distance = 1024,
                       norm_init_var = 1.0,
                       norm_bias = True,
                       norm_eps = 1e-5,
                       att_init_mean = 0.0,
                       att_init_std = 0.02,
                       att_bias = True,
                       att_mask_value = float("-1e4"),
                       ffn_init_mean = 0.0,
                       ffn_init_std = 0.02,
                       ffn_bias = True,
                       ffn_activate_fn = "gelu",
                       proj_init_mean = 0.0,
                       proj_init_std = 1,
                       proj_bias = True,
                       length_scale = False,
                       attn_scale = True,
                       half = True,
                       int8 = False,
                       tied = True,
                       cls_head = None,
                    ):

        super().__init__()

        self.vocab_size = vocab_size
        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.num_layers = num_layers
        self.dropout_p = dropout_p
        self.emb_init_mean = emb_init_mean
        self.emb_init_std = emb_init_std
        self.pos_bias_type = pos_bias_type
        self.position_bias_max_distance = position_bias_max_distance
        self.norm_init_var = norm_init_var
        self.norm_bias = norm_bias
        self.norm_eps = norm_eps
        self.att_init_mean = att_init_mean
        self.att_init_std = att_init_std
        self.att_bias = att_bias
        self.att_mask_value = att_mask_value
        self.ffn_init_mean = ffn_init_mean
        self.ffn_init_std = ffn_init_std
        self.ffn_bias = ffn_bias
        self.ffn_activate_fn = ffn_activate_fn
        self.proj_init_mean = proj_init_mean
        self.proj_init_std = proj_init_std
        self.proj_bias = proj_bias
        self.length_scale = length_scale
        self.attn_scale = attn_scale
        self.int8 = int8
        self.tied = tied
        if half: 
            self.dtype = torch.half
        else:
            self.dtype = torch.float
        self.cls_head = cls_head