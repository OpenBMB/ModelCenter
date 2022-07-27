# coding=utf-8
# Copyright 2022 The OpenBMB team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from ..layer import Encoder, Decoder, Embedding, Linear, RelativePositionEmbedding
from .basemodel import BaseModel
from .config import CPM2Config
from transformers.modeling_outputs import Seq2SeqModelOutput


class CPM2(BaseModel):

    _CONFIG_TYPE = CPM2Config

    def __init__(self, config: CPM2Config):
        
        super().__init__()
        
        self.config = config

        self.encoder = Encoder(
            num_layers = config.num_encoder_layers,
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

        self.decoder = Decoder(
            num_layers = config.num_decoder_layers,
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
            length_scale = False, # TODO not an elegent implementation # config.length_scale,
            dtype = config.dtype,
            int8 = config.int8,
            init_mean = config.emb_init_mean,
            init_std = config.emb_init_std,
        )

        self.position_bias_enc = RelativePositionEmbedding(
            num_heads = config.num_heads, 
            num_buckets = config.position_bias_num_buckets, 
            max_distance = config.position_bias_max_distance, 
            bidirectional = True, 
            dtype = config.dtype,
            init_mean = config.pos_init_mean,
            init_std = config.pos_init_std,
        )

        self.position_bias_dec = RelativePositionEmbedding(
            num_heads = config.num_heads, 
            num_buckets = config.position_bias_num_buckets, 
            max_distance = config.position_bias_max_distance, 
            bidirectional = False, 
            dtype = config.dtype,
            init_mean = config.pos_init_mean,
            init_std = config.pos_init_std,
        )

        if self.config.cls_head:
            self.cls_projection = Linear(
                dim_out = self.config.cls_head,
                dim_in = config.dim_model,
                length_scale = config.length_scale,
                dtype = config.dtype,
                int8 = config.int8,
                init_mean = config.proj_init_mean,
                init_std = config.proj_init_std,
                bias = config.proj_bias,
            )

        if not self.config.tied:
            self.output_projection = Linear(
                dim_out = config.vocab_size,
                dim_in = config.dim_model,
                length_scale = config.length_scale,
                dtype = config.dtype,
                int8 = config.int8,
                init_mean = config.proj_init_mean,
                init_std = config.proj_init_std,
                bias = config.proj_bias,
            )

    def forward(self, 
                input_ids = None, # (batch, seq_enc)
                length = None, # (batch)
                decoder_input_ids = None, # (batch, seq_dec)
                decoder_length = None, # (batch)
                attention_mask = None, # (batch, seq_enc)
                decoder_attention_mask = None, # (batch, seq_dec)
                head_mask = None, # unused
                decoder_head_mask = None, # unused
                cross_attn_head_mask = None, # unused
                encoder_outputs = None,
                inputs_embeds = None, 
                decoder_inputs_embeds = None,
                use_cache=False,
                past_key_values=None,
                output_attentions = None, # unused
                output_hidden_states = None, # unused
                return_dict = True,
                return_logits = False,
        ):
        """ 
        CPM-2 is an encoder-decoder model and converts problems into a text-to-text format.
        This model inherits from BaseModel. This model is also a PyTorch torch.nn.Module subclass. You can use it as a regular PyTorch Module.
        You can also select the data and data type that you want the model to return through changing the value of `return_dict` and `return_logits`.

        Args:
            input_ids (:obj:`torch.Tensor` of shape ``(batch, seq_enc)``): Indices of input sequence tokens. It will be embedded by model's internal embedding lookup matrix.
            length (:obj:`torch.Tensor` of shape ``(batch)``): Length of input sequence before padding.  
            attention_mask (:obj:`torch.Tensor` of shape ``(batch, seq_enc)``): Used to avoid performing attention on padding token indices in input.
            decoder_input_ids (:obj:`torch.Tensor` of shape ``(batch, seq_enc)``): Indices of decoder input sequence tokens .
            decoder_length (:obj:`torch.Tensor` of shape ``(batch)``): Length of decoder input sequence before padding.
            deocoder_attention_mask (:obj:`torch.Tensor` of shape ``(batch, seq_enc)``): Used to avoid performing attention on padding token indices in decoder input.
            head_mask (:obj:`torch.Tensor` of shape ``(num_layers, num_heads)``): Unused.
            decoder_head_mask (:obj:`torch.Tensor` of shape ``(num_layers, num_heads)``): Unused.
            cross_attn_head_mask (:obj:`torch.Tensor` of shape ``(num_layers, num_heads)``): Unused.
            encoder_outputs (:obj:`torch.Tensor` of shape ``(batch, dim_model, seq_enc)``): Outputs of encoder. 
            inputs_embeds (:obj:`torch.Tensor` of shape ``(batch, seq_enc, dim_model)``): Embedding of the input. You can choose to directly pass the inputs embedding to control the way of embedding. 
            decoder_inputs_embeds (:obj:`torch.Tensor` of shape ``(batch, seq_dec, dim_model)``): Embedding of the decoder input. You can choose to directly pass the inputs embedding to control the way of embedding. 
            output_attentions (:obj:`torch.Tensor` of shape ``(batch, num_heads, seq_enc, seq_enc)``): Unused.
            output_hidden_states (:obj:`torch.Tensor` of shape ``(batch, seq_dec, dim_model)``): Unused.
            return_dict (:obj:`bool`): Whether to return a Seq2SeqModelOutput instead of just a tuple.
            return_logits (:obj:`bool`): Whether to return the prediction score for each token in vocabulary (before softmax).

        Return:
            Seq2SeqModelOutput or tuple or torch.Tensor of shape (batch, seq_dec, vocab_output_size) or (batch, seqlen, cls_head): The T5 output. Depended on the value of `return_dict` and `return_logits` 
        """
        # encoder
        if encoder_outputs is None:
            assert input_ids is not None or inputs_embeds is not None

            if input_ids is not None:
                batch = input_ids.size(0)
                seq_enc = input_ids.size(1)
                device = input_ids.device
            else:
                batch = inputs_embeds.size(0)
                seq_enc = inputs_embeds.size(1)
                device = inputs_embeds.device
            
            with torch.no_grad():
                if attention_mask is not None:
                    attention_mask = attention_mask.to(torch.bool)
                else:
                    attention_mask = torch.arange(seq_enc, device=device)[None, :].repeat(batch, 1) < length[:, None]
                # (batch, seq_enc, seq_enc)
                enc_attention_mask = attention_mask.view(batch, seq_enc, 1) & attention_mask.view(batch, 1, seq_enc)

            # (num_heads, seq_enc, seq_enc)
            enc_position_bias = self.position_bias_enc(seq_enc, seq_enc)
            
            # (batch, dim_model, seq_enc)
            if inputs_embeds is None:
                hidden_states_enc = self.input_embedding(input_ids)
            else:
                hidden_states_enc = inputs_embeds

            # (batch, dim_model, seq_enc)
            encoder_outputs = self.encoder(hidden_states_enc, enc_attention_mask, enc_position_bias)
        else:
            seq_enc = encoder_outputs.size(1)

        # decoder
        assert decoder_input_ids is not None or decoder_inputs_embeds is not None

        if decoder_input_ids is not None:
            batch = decoder_input_ids.size(0)
            seq_dec = decoder_input_ids.size(1)
            device = decoder_input_ids.device
        else:
            batch = decoder_inputs_embeds.size(0)
            seq_dec = decoder_inputs_embeds.size(1)
            device = decoder_inputs_embeds.device

        pkv_len = 0 if past_key_values is None else past_key_values[0][0].size(-2)
        seq_length = pkv_len + seq_dec
        with torch.no_grad():
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(torch.bool)
            else:
                decoder_attention_mask = torch.arange(seq_length, device=device)[None, :].repeat(batch, 1) < decoder_length[:, None]
            directional_mask_2d = torch.arange(seq_length, device=device) <= torch.arange(seq_length, device=device).view(-1, 1)
            dec_attention_mask = decoder_attention_mask.view(batch, seq_length, 1) & decoder_attention_mask.view(batch, 1, seq_length) & directional_mask_2d.view(1, seq_length, seq_length)
            cross_attention_mask = attention_mask.view(batch, 1, seq_enc) & decoder_attention_mask.view(batch, seq_length, 1)

        dec_position_bias = self.position_bias_dec(seq_length, seq_length)
        dec_attention_mask = dec_attention_mask[:, -seq_dec:, :]
        dec_position_bias = dec_position_bias[:, -seq_dec:, :]
        cross_attention_mask = cross_attention_mask[:,-seq_dec:,:]

        if decoder_inputs_embeds is None:
            hidden_states_dec = self.input_embedding(decoder_input_ids)
        else:
            hidden_states_dec = decoder_inputs_embeds

        current_key_values = None
        if use_cache:
            decoder_outputs, current_key_values=self.decoder(hidden_states_dec, dec_attention_mask, dec_position_bias, 
                                                             encoder_outputs, cross_attention_mask, None,
                                                             use_cache = use_cache, past_key_values = past_key_values)
        else:
            decoder_outputs = self.decoder(hidden_states_dec, dec_attention_mask, dec_position_bias,
                                           encoder_outputs, cross_attention_mask, None)

        logits = None
        if return_logits:
            if self.config.cls_head:
                logits = self.cls_projection(decoder_outputs)
            elif self.config.tied:
                logits = self.input_embedding.projection(decoder_outputs)
            elif not self.config.tied:
                logits = self.output_projection(decoder_outputs)
            return logits

        if not return_dict:
            return cross_attention_mask, encoder_outputs, current_key_values, None, None, None, None, None
        else:
            return Seq2SeqModelOutput(
                last_hidden_state=decoder_outputs,
                encoder_last_hidden_state=encoder_outputs,
                past_key_values=current_key_values,
                encoder_hidden_states=None,
                decoder_hidden_states=None,
                decoder_attentions=None,
                cross_attentions=None,
                encoder_attentions=None,
            )