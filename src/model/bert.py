import torch

from torch import Tensor, device
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union        

from layer import Encoder, Embedding, Linear
from model.config import BertConfig
from model.basemodel import BaseModel

from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

class BertPooler(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = Linear(hidden_size, hidden_size, bias=True)
        self.activation = torch.nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, :, 0:1].repeat(1,1,2).contiguous()
        pooled_output = self.dense(first_token_tensor)[:, :, 0]
        pooled_output = self.activation(pooled_output)
        return pooled_output
    

class Bert(BaseModel):
    _CONFIG_TYPE = BertConfig
    
    def __init__(self, config: BertConfig):
        super().__init__()
        self.word_embeddings = Embedding(
            config.vocab_size,
            config.hidden_size
        )
        self.position_embeddings = Embedding(
            config.max_position_embeddings, 
            config.hidden_size
        )
        self.token_type_embeddings = Embedding(
            config.type_vocab_size, 
            config.hidden_size
        )
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

        self.encoder = Encoder(
            num_layers=config.num_hidden_layers,
            dim_model=config.hidden_size,
            dim_ff=config.intermediate_size,
            num_heads=config.num_attention_heads,
            dim_head=int(config.hidden_size / config.num_attention_heads),
            #dropout_p=config.attention_probs_dropout_prob, TODO
            attn_scale=True,
            att_bias=True,
            norm_bias=True,
            norm_eps=config.layer_norm_eps,
            post_layer_norm=True,
            ffn_activate_fn="gelu",
            ffn_bias=True,
        )

        self.pooler = BertPooler(config.hidden_size)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None, #unused
        inputs_embeds=None,
        encoder_hidden_states=None, #unused
        encoder_attention_mask=None, #unused
        output_attentions=None, #unused
        output_hidden_states=None, #unused
        return_dict=True,
    ):
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        inputs_embeds = self.word_embeddings(input_ids.to(torch.int32))
        token_type_embeddings = self.token_type_embeddings(token_type_ids.to(torch.int32))
        position_embeddings = self.position_embeddings(position_ids.to(torch.int32))
        embedding_output = inputs_embeds + token_type_embeddings + position_embeddings

        attention_mask = attention_mask.to(torch.bool)
        attention_mask = attention_mask.view(batch_size, seq_length, 1).repeat(1, 1, seq_length)

        sequence_output  = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
        )

        pooled_output = self.pooler(sequence_output)

        if not return_dict:
            return (sequence_output, pooled_output)
        else:
            return BaseModelOutputWithPoolingAndCrossAttentions(
                last_hidden_state=sequence_output,
                pooler_output=pooled_output,
                past_key_values=None,
                hidden_states=None,
                attentions=None,
                cross_attentions=None,
            )