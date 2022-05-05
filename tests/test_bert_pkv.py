#coding:utf-8

import torch
import bmtrain as bmt

from model_center.tokenizer import BertTokenizer
from model_center.model import BertConfig, Bert

from transformers import BertModel as hugBert

def main():
    bmt.init_distributed()

    path = "bert-base-uncased"
    test_hug_bert = False #和bert对拍需要修改hugbert的代码
    config = BertConfig.from_pretrained(path)
    config.dropout_p = 0
    bmt_bert = Bert.from_pretrained(path, config=config)
    if test_hug_bert:
        hug_bert = hugBert.from_pretrained(path).cuda().eval().half()

    cur_len = 0
    add_len = 32
    bmt_pkv = None
    hug_pkv = None

    input_ids_list = []
    logits_list = []
    attention_mask_all = None

    for _ in range(10):
        batch = 2
        input_ids = torch.randint(config.vocab_size, (batch, add_len,), dtype=torch.int32).cuda()
        attention_mask = torch.randint(2, (batch, add_len, add_len + cur_len), dtype=torch.int32).cuda()

        bmt_res = bmt_bert(input_ids = input_ids, attention_mask = attention_mask, use_cache = True, past_key_values = bmt_pkv)
        bmt_pkv = bmt_res.past_key_values
        bmt_logits = bmt_res.last_hidden_state
        if test_hug_bert:
            hug_res = hug_bert(input_ids = input_ids, attention_mask = attention_mask, use_cache = True, past_key_values = hug_pkv)
            hug_pkv = hug_res.past_key_values
            hug_logits = hug_res.last_hidden_state

            d = 0
            siz = (-1,) + bmt_pkv[0][0].size()[-2:]
            for i in range(len(hug_pkv)):
                for j in [0, 1]:
                    d += (hug_pkv[i][j].contiguous().view(siz) - bmt_pkv[i][j]).abs().max()
            print(d)
            print((hug_logits - bmt_logits).abs().max())

        input_ids_list.append(input_ids)
        logits_list.append(bmt_logits)
        if attention_mask_all is None:
            attention_mask_all = attention_mask
        else:
            attention_mask_all = torch.cat([attention_mask_all, torch.zeros(batch, cur_len, add_len).cuda()], dim=2)
            attention_mask_all = torch.cat([attention_mask_all, attention_mask], dim=1)

        cur_len += add_len

    input_ids = torch.cat(input_ids_list, dim=1)
    logits_pkv = torch.cat(logits_list, dim=1)
    logits = bmt_bert(input_ids = input_ids, attention_mask = attention_mask_all).last_hidden_state
    print((logits - logits_pkv).abs().max())
    if test_hug_bert:
        logits = hug_bert(input_ids = input_ids, attention_mask = attention_mask_all).last_hidden_state
        print((logits - logits_pkv).abs().max())

if __name__ == "__main__":
    main()
