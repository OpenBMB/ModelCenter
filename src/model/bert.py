import torch

from torch import Tensor, device
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union        

from layer.bert_layers import BertEncoder, BertEmbeddings, BertPooler

from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from . import BertConfig

def get_parameter_dtype(parameter):
    try:
        return next(parameter.parameters()).dtype
    except StopIteration:
        # For nn.DataParallel compatibility in PyTorch 1.5

        def find_tensor_attributes(module: torch.nn.Module) -> List[Tuple[str, Tensor]]:
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
            return tuples

        gen = parameter._named_members(get_members_fn=find_tensor_attributes)
        first_tuple = next(gen)
        return first_tuple[1].dtype

class Bert(torch.nn.Module):
    def __init__(self, config : BertConfig,  add_pooling_layer=True):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)
        
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
        past_key_values=None, #delete
        use_cache=None, #delete
        output_attentions=None, #unused
        output_hidden_states=None, #unused
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        attention_mask = attention_mask.to(torch.bool)
        attention_mask = attention_mask.view(batch_size, seq_length, 1).repeat(1, 1, seq_length)
        #attention_mask = attention_mask.view(batch_size, seq_length, 1) & attention_mask.view(batch_size, 1, seq_length)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )


        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            return_dict=return_dict,
        )

        #return {'last_hidden_state' : encoder_outputs}

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )