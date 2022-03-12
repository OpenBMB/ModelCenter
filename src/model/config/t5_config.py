import torch
from .config import Config

class T5Config(Config):

    def __init__(self, dim_model=768,
                       num_heads=12,
                       dim_head=64,
                       dim_ff=3072,
                       num_encoder_layers=12,
                       num_decoder_layers=12,
                       dropout_p=0,
                       emb_init_mean = 0.0,
                       emb_init_std = 1,
                       pos_bias_type = "relative",
                       position_bias_num_buckets=32,
                       position_bias_max_distance=128,
                       pos_init_mean = 0.0,
                       pos_init_std = 1,
                       norm_init_var = 1.0,
                       norm_bias = False,
                       norm_eps = 1e-6,
                       att_init_mean = 0.0,
                       att_init_std = 1,
                       att_bias = False,
                       att_mask_value = -1e5,
                       ffn_init_mean = 0.0,
                       ffn_init_std = 1,
                       ffn_bias = False,
                       ffn_activate_fn = "relu",
                       proj_init_mean = 0.0,
                       proj_init_std = 1,
                       proj_bias = False,
                       length_scale = False, 
                       attn_scale = False,
                       half = True,
                       int8 = False,
                       cls_head = None,
                    ):

        super().__init__()

        self.dim_model = dim_model
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.dim_ff = dim_ff
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dropout_p = dropout_p
        self.emb_init_mean = emb_init_mean
        self.emb_init_std = emb_init_std
        self.pos_bias_type = pos_bias_type
        self.position_bias_num_buckets = position_bias_num_buckets
        self.position_bias_max_distance = position_bias_max_distance
        self.pos_init_mean = pos_init_mean
        self.pos_init_std = pos_init_std
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
        if half: 
            self.dtype = torch.half
        else:
            self.dtype = torch.float
        self.cls_head = cls_head