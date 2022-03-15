#coding:utf-8

import os
import torch
import bmtrain as bmt

from model_center.tokenizer import BertTokenizer
from model_center.model import BertConfig, Bert
from model_center.arguments import get_args

from transformers import BertModel

def get_tokenizer(args):
    ver = args.model_config.split('/')[-1]
    print('ver =', ver)
    return BertTokenizer.from_pretrained(ver)

def get_model(args):
    config = BertConfig.from_pretrained(args.model_config)
    print(config)
    model = Bert(config)
    bmt.load(model, args.load)#'/home/hx/lyq/workshop/save_model')
    return model

def main():
    args = get_args()
    bmt.init_distributed()

    version = args.model_config.split('/')[-1]
    print('version = ', version, ' | ', args.model_config)

    device = 'cuda:0'

    tokenizer = get_tokenizer(args)
    bert = BertModel.from_pretrained(version).to(device).half()
    fake_bert = get_model(args).to(device)

    bert.eval()
    fake_bert.eval()

    logits = tokenizer([
        "这怎么这么菜啊",
        "才2.9万啊，我还以为9.2万呢",
        #"i like apple."
        #"water is difficult"
        ], 
        max_length=64,
        return_tensors='pt', 
        padding='max_length',
        truncation=True,
    ).to(device)

    batch_size, seq_length = logits['input_ids'].size()
    mask = logits['attention_mask']
    dim_model = BertConfig.from_pretrained(args.model_config).dim_model
    mask = mask.view(batch_size, seq_length, 1).repeat(1, 1, dim_model)

    x = bert(**logits)['last_hidden_state'] * mask
    fx = fake_bert(**logits)['last_hidden_state'].transpose(1,2) * mask

    print(fx,x,sep='\n')
    print(torch.abs(fx - x).max() / torch.abs(x).max())
    print((torch.abs(fx - x)).max())
    print(torch.abs(fx - x).sum() / x.numel())
    print((torch.abs(fx - x) > 0.1).sum())

if __name__ == "__main__":
    main()
