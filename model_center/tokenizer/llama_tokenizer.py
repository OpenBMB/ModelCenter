# coding=utf-8

# The current implementation is mainly to adapt the training framework of the Transformers toolkit, 
# and replace the original model implementation.
# TODO we will change to our SAM implementation in the future, which will be a more efficient tokenizer

from .base_tokenizer import BaseTokenizer
from transformers import LlamaTokenizer as LlamaTokenizerTransformers

class LlamaTokenizerBase(BaseTokenizer):
    def from_pretrained(self, pretrained_model_name_or_path, *args, **kwargs):
        tokenizer = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        return tokenizer

LlamaTokenizer = LlamaTokenizerBase(LlamaTokenizerTransformers)
