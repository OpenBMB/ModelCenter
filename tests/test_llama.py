#coding:utf-8

import torch
import bmtrain as bmt
from model_center.model.config import LlamaConfig
from model_center.model import Llama
from model_center.tokenizer import LlamaTokenizer

from transformers import LlamaForCausalLM
from transformers import LlamaTokenizer as LlamaTokenizerHF

def main():
    # path = f"../results/llama-7b"
    # hf_path = f"../results/llama-7b-hf"
    path = f"../results/llama-2-7b"
    hf_path = f"../results/llama-2-7b-hf"
    
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

def generate():
    # only one GPU is enough
    from model_center.generation.llama import LlamaBeamSearch, LlamaRandomSampling
    path = f"../results/llama-7b"
    
    tokenizer = LlamaTokenizer.from_pretrained(path)
    model = Llama.from_pretrained(path)

    beam_search = LlamaBeamSearch(
        model=model,
        tokenizer=tokenizer,
    )
    random_search = LlamaRandomSampling(
        model=model,
        tokenizer=tokenizer,
    )

    data_list = [
        "Beijing is the capital of",
        "Steven Jobs",
    ]

    inference_results = beam_search.generate(data_list, max_length=100)
    print("beam search:")
    for res in inference_results:
        print(res)
    print("random sampling:")
    inference_results = random_search.generate(data_list, max_length=100)
    for res in inference_results:
        print(res)

if __name__ == "__main__":
    bmt.init_distributed(seed=2333)
    main()
    # generate()
