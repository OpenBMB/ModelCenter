
#coding:utf-8

import time
import torch
import bmtrain as bmt

from model_center.tokenizer import T5Tokenizer
from model_center.model import T5Config, T5
from transformers import T5ForConditionalGeneration as hugT5


def main():
    bmt.init_distributed()

    # path = "bert-base-uncased"
    # config = BertConfig.from_pretrained(path)
    # config.dropout_p = 0
    # bmt_bert = Bert.from_pretrained(path, config=config)
    # hug_bert = hugBert.from_pretrained(path).cuda().eval().half()

    path = "t5-base"
    config = T5Config.from_pretrained(path)
    config.dropout_p = 0
    bmt_t5 = T5.from_pretrained(path, config=config)
    config = T5Config.from_pretrained(path, use_cache = True)
    cac_t5 = T5.from_pretrained(path, config=config)

    cur_len = 0
    add_len = 1
    cac_pkv = None

    input_ids_list = []
    dec_input_ids_list = []
    bmt_logits_list = []
    cac_logits_list = []

    for idx in range(50):
        print(idx)

        batch = 1
        max_encoder_length = 512
        max_decoder_length = 512
        input_ids = torch.randint(config.vocab_size, (batch, max_encoder_length,), dtype=torch.int32).cuda()
        length = torch.randint(max_encoder_length, (batch, ), dtype=torch.int32).cuda()
        decoder_input_ids = torch.randint(config.vocab_size, (batch, max_decoder_length,), dtype=torch.int32).cuda()
        decoder_length = torch.randint(max_decoder_length, (batch, ), dtype=torch.int32).cuda()
        attention_mask = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :].repeat(input_ids.shape[0], 1) < length[:, None]
        decoder_attention_mask = torch.arange(decoder_input_ids.shape[1], device=decoder_input_ids.device)[None, :].repeat(decoder_input_ids.shape[0], 1) < decoder_length[:, None]
        cac_pkv = None
        
        input_ids_list.append(input_ids)
        dec_input_ids_list.append(decoder_input_ids)

        bmt_res = bmt_t5(input_ids = input_ids,length=length, decoder_input_ids = decoder_input_ids, decoder_length=decoder_length, attention_mask=attention_mask, decoder_attention_mask=decoder_attention_mask, return_logits=True)
        bmt_logits = bmt_res[0]
        bmt_logits_list.append(bmt_logits)

        cac_res = cac_t5(input_ids = input_ids,length=length, decoder_input_ids = decoder_input_ids, decoder_length=decoder_length, attention_mask=attention_mask, decoder_attention_mask=decoder_attention_mask, use_cache = True, past_key_values = cac_pkv, return_logits=True)
        cac_pkv = cac_res[2]
        cac_logits = cac_res[0]
        cac_logits_list.append(cac_logits)

        cur_len += add_len
    
    
    bmt_logits_pkv = torch.cat(bmt_logits_list, dim=1)
    cac_logits_pkv = torch.cat(cac_logits_list, dim=1)
    print((bmt_logits_pkv - cac_logits_pkv).abs().mean())

    input_ids = torch.cat(input_ids_list, dim=1)
    decoder_input_ids = torch.cat(dec_input_ids_list, dim=1)
    logits = bmt_t5(input_ids = input_ids,length=length, decoder_input_ids = decoder_input_ids,decoder_length=decoder_length,  attention_mask = torch.ones((2, cur_len), dtype=torch.int32).cuda(), return_logits=True)[0]
    print((logits - bmt_logits_pkv).abs().mean())

if __name__ == "__main__":
    main()