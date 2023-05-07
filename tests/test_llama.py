#coding:utf-8

import torch
import bmtrain as bmt
from model_center.model.config import LlamaConfig
from model_center.model import Llama
from model_center.tokenizer import LlamaTokenizer

from transformers import LlamaForCausalLM
from transformers import LlamaTokenizer as LlamaTokenizerHF

def main():
    bmt.init_distributed(seed=2333)
    path = f"../results/llama-7b"
    hf_path = f"../results/llama-7b-hf"
    
    tokenizer = LlamaTokenizer.from_pretrained(path)
    config = LlamaConfig.from_pretrained(path)
    bmt_llama = Llama.from_pretrained(path, config=config)   
    hug_llama = LlamaForCausalLM.from_pretrained(hf_path).half().eval().cuda()

    for ith in range(1, 11):
        batch = 1
        max_encoder_length = ith * 16
        input_ids = torch.randint(config.vocab_size, (batch, max_encoder_length,), dtype=torch.int32).cuda()
        length = torch.randint(max_encoder_length, (batch, ), dtype=torch.int32).cuda()
        attention_mask = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :].repeat(input_ids.shape[0], 1) < length[:, None]

        bmt_logits = bmt_llama(input_ids = input_ids, attention_mask = attention_mask, output_logits=True).logits
        hug_logits = hug_llama(input_ids = input_ids, attention_mask = attention_mask).logits
        b = bmt_logits*attention_mask[:,:,None]
        h = hug_logits*attention_mask[:,:,None]
        d = (h - b).abs()
        if bmt.rank() == 0:
            print(d.max())

if __name__ == "__main__":
    main()
