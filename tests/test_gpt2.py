#coding:utf-8

import torch
import bmtrain as bmt

from model_center.tokenizer import GPT2Tokenizer
from model_center.model import GPT2Config, GPT2

from transformers import GPT2LMHeadModel as hugGPT2

def main():
    bmt.init_distributed()

    path = "gpt2-base"
    tokenizer = GPT2Tokenizer.from_pretrained(path)
    config = GPT2Config.from_pretrained(path)
    config.dropout_p = 0
    bmt_gpt2 = GPT2.from_pretrained(path, config=config)

    hug_gpt2 = hugGPT2.from_pretrained('gpt2').cuda().eval().half()
    
    for _ in range(10):
        batch = 1
        max_encoder_length = 512
        input_ids = torch.randint(config.vocab_size, (batch, max_encoder_length,), dtype=torch.int32).cuda()
        length = torch.randint(max_encoder_length, (batch, ), dtype=torch.int32).cuda()
        attention_mask = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :].repeat(input_ids.shape[0], 1) < length[:, None]

        bmt_logits = bmt_gpt2(input_ids = input_ids, attention_mask = attention_mask, return_logits=True)
        hug_logits = hug_gpt2(input_ids = input_ids, attention_mask = attention_mask).logits
        b = (bmt_logits*attention_mask[:,:,None])[:,:,:-1]
        h = hug_logits*attention_mask[:,:,None]
        d = (h - b).abs()
        print(d.max())

if __name__ == "__main__":
    main()
