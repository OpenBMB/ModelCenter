import torch
from layer import Encoder, Decoder, Embedding, Projection, RelativePositionEmbedding, RotaryEmbedding
from layer import LayerNorm
import bmtrain as bmp
import cpm_kernels.torch as ct

class GPTj(torch.nn.Module):
    
    def __init__(self, config):
        
        super().__init__()

        self.decoder = Encoder(
            num_layers = config.num_layers,
            dim_model = config.dim_model, 
            dim_ff = config.dim_ff,
            num_heads = config.num_heads,
            dim_head = config.dim_head,
            dtype = config.dtype, 
            int8 = config.int8,
            norm_eps = config.norm_eps, 
            norm_init_var = config.norm_init_var,
            norm_bias = config.norm_bias,
            att_init_mean = config.att_init_mean, 
            att_init_std = config.att_init_std,
            att_bias = config.att_bias,
            att_mask_value = float(config.att_mask_value),
            pos_bias_type = config.pos_bias_type,
            ffn_init_mean = config.ffn_init_mean, 
            ffn_init_std = config.ffn_init_std,
            ffn_bias = config.ffn_bias,
            ffn_activate_fn = config.ffn_activate_fn,
            length_scale = config.length_scale,
            attn_scale = config.attn_scale,
            dropout_p = config.dropout_p,
            parallel_ffn = True,
        )

        self.input_embedding = Embedding(
            vocab_size = config.vocab_size, 
            embedding_size = config.dim_model,
            length_scale = config.length_scale,
            dtype = config.dtype,
            int8 = config.int8,
            init_mean = config.emb_init_mean,
            init_std = config.emb_init_std,
        )

        self.position_bias = RotaryEmbedding(
            rotary_dim = config.pos_rotary_dim,
        )
        
        self.tied = config.tied
        self.cls_head = config.cls_head
        if self.cls_head or not self.tied:
            self.output_projection = Projection(
                dim_out = self.cls_head if self.cls_head else config.vocab_size,
                dim_in = config.dim_model,
                length_scale = config.length_scale,
                dtype = config.dtype,
                int8 = config.int8,
                init_mean = config.proj_init_mean,
                init_std = config.proj_init_std,
                bias = config.proj_bias,
            )

    def forward(self, input_ids : torch.Tensor, # (batch, seqlen)
                      length : torch.Tensor, # (batch)
    ):

        batch = input_ids.size(0)
        seq_dec = input_ids.size(1)
        device = input_ids.device

        with torch.no_grad():
            dec_mask_1d = torch.arange(seq_dec, device=device)[None, :].repeat(batch, 1) < length[:, None]
            directional_mask_2d = torch.arange(seq_dec, device=device).view(-1, 1) <= torch.arange(seq_dec, device=device)
            dec_attention_mask = dec_mask_1d.view(batch, seq_dec, 1) & directional_mask_2d.view(1, seq_dec, seq_dec)

        hidden_states = self.input_embedding(input_ids)

        hidden_states = self.decoder(hidden_states, dec_attention_mask, self.position_bias)

        logits = self.output_projection(hidden_states)

        return logits
