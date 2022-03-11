import torch
from layer import Encoder, Decoder, Embedding, Projection, RelativePositionEmbedding
from layer import LayerNorm
import bmtrain as bmt
import cpm_kernels.torch as ct

class GPT2(torch.nn.Module):
    
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

        self.position_embedding = Embedding(
            vocab_size = config.position_bias_max_distance, 
            embedding_size = config.dim_model,
            length_scale = config.length_scale,
            dtype = config.dtype,
            int8 = config.int8,
            init_mean = config.emb_init_mean,
            init_std = config.emb_init_std,
        )

        self.embed_dropout = torch.nn.Dropout(config.dropout_p)
        
        self.tied = config.tied
        self.cls_head = config.cls_head
        if self.cls_head:
            self.output_projection = Projection(
                dim_out = self.cls_head,
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
            position_ids = torch.arange(seq_dec, dtype=torch.int32, device=device)[None, :].repeat(batch, 1)

            dec_mask_1d = torch.arange(seq_dec, device=device)[None, :].repeat(batch, 1) < length[:, None]
            directional_mask_2d = torch.arange(seq_dec, device=device).view(-1, 1) <= torch.arange(seq_dec, device=device)
            dec_attention_mask = dec_mask_1d.view(batch, seq_dec, 1) & dec_mask_1d.view(batch, 1, seq_dec) & directional_mask_2d.view(1, seq_dec, seq_dec)

        hidden_states = self.input_embedding(input_ids)

        position_embeds = self.position_embedding(position_ids)
        hidden_states = ct.element_add(hidden_states, position_embeds)

        hidden_states = self.embed_dropout(hidden_states)

        hidden_states = self.decoder(hidden_states, dec_attention_mask)

        if self.cls_head:
            logits = self.output_projection(hidden_states)
        else:
            logits = self.input_embedding.projection(hidden_states)
            logits[:, :, -1] = -float("inf") # TODO not an elegant implementation, gpt2 vocab is odd number, expand to even and ignore last

        return logits
