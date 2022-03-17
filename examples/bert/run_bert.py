#coding:utf-8

import os
import torch
import bmtrain as bmt
import IPython

from model_center.tokenizer import BertTokenizer
from model_center.model import BertConfig, Bert
from model_center.arguments import get_args

from transformers import BertModel, BertLMHeadModel

def get_tokenizer(args):
    return BertTokenizer.from_pretrained(args.model_config)

def get_model(args):
    return Bert.from_pretrained(args.model_config)

def main():
    args = get_args()
    bmt.init_distributed()

    version = args.model_config
    print('version = ', version)

    device = 'cuda:0'

    tokenizer = get_tokenizer(args)
    #bert : BertModel = BertModel.from_pretrained(version).to(device).half()
    bert : BertLMHeadModel = BertLMHeadModel.from_pretrained(version).to(device).half()
    fake_bert : Bert = get_model(args).to(device)

    bert.eval()
    fake_bert.eval()

    logits = tokenizer([
        "i like apple.",
        "water is difficult",
        ], 
        max_length=64,
        return_tensors='pt', 
        padding='max_length',
        truncation=True,
    ).to(device)

    batch_size, seq_length = logits['input_ids'].size()
    vocab_size = max(fake_bert.input_embedding.weight.size())
    mask = logits['attention_mask']
    #dim_model = BertConfig.from_pretrained(args.model_config).dim_model
    mask = mask.view(batch_size, seq_length, 1).repeat(1, 1, vocab_size)

    x = bert(**logits)['logits'] * mask
    fx = fake_bert(**logits, return_logits=True).transpose(1,2) * mask

    print(fx,x,sep='\n')
    print((torch.abs(fx - x)).max())

if __name__ == "__main__":
    main()
