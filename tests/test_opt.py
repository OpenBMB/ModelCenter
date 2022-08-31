#coding:utf-8

import torch
import bmtrain as bmt

from model_center.tokenizer import OPTTokenizer
from model_center.model import OPTConfig, OPT
from transformers import OPTForCausalLM as hugOPT

def main():
    bmt.init_distributed()

    ver = "2.7b"
    path = f"opt-{ver}"
    tokenizer = OPTTokenizer.from_pretrained(path)
    config = OPTConfig.from_pretrained(path)
    config.dropout_p = 0
    bmt_opt = OPT.from_pretrained(path, config=config)

    hug_opt = hugOPT.from_pretrained(f'opt-{ver}').cuda().eval().half()
    def hook(name):
        def backward_hook(module, grad_input, grad_output):
            emb_grad[name]=grad_output[0]
        return backward_hook
    emb_grad={}
    for i in range(10):
        batch = 1
        max_encoder_length = 512
        input_ids = torch.randint(config.vocab_size, (batch, max_encoder_length,), dtype=torch.int32).cuda()
        length = torch.randint(max_encoder_length, (batch, ), dtype=torch.int32).cuda()
        attention_mask = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :].repeat(input_ids.shape[0], 1) < length[:, None]
        
        bmt_logits = bmt_opt(input_ids = input_ids, attention_mask = attention_mask, output_logits=True).logits
        hug_logits = hug_opt(input_ids = input_ids, attention_mask = attention_mask).logits
        b = bmt_logits*attention_mask[:,:,None]
        h = hug_logits*attention_mask[:,:,None]
        d = (h - b).abs()
        print(d.max())
        if i == 0:
            b_emb=bmt_opt._modules['input_embedding']
            h_emb=hug_opt._modules['model']._modules['decoder']._modules['embed_tokens']
            h_emb.register_full_backward_hook(hook("h"))
            b_emb.register_full_backward_hook(hook("b"))
        else:
            emb_grad.clear()
        loss_func = torch.nn.CrossEntropyLoss()
        labels = torch.randint(config.vocab_size, (batch, max_encoder_length,), dtype=torch.long).cuda()
        loss1 = loss_func(b.view(-1,b.shape[-1]), labels.view(-1))
        loss2 = loss_func(h.view(-1,h.shape[-1]), labels.view(-1))
        loss1.backward()
        loss2.backward()
        if i>0:
            d_grad=(emb_grad["h"]-emb_grad["b"]).abs()
            print(d_grad.max())
if __name__ == "__main__":
    main()
