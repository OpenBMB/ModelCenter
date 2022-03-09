#coding:utf-8

import time
import random
import torch
import bmpretrain as bmp
import numpy as np
import os

from torch import Tensor, device
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union        

from tokenizer import BertTokenizer
from model import BertConfig, Bert

from arguments import get_args
from generation import generate_no_beam

from transformers import BertModel

def pl():
    print('')
    print('========')
    print('')

def main():
    bmp.init_distributed()
    device = 'cuda:0'

    #x = torch.rand(2,4).to(device).to(torch.half)
    #y = torch.rand(4,6).to(device).to(torch.half)
    #p = Projection(2,4, weight=x)
    #print(p(y[None,:,:]))
    #print(x.matmul(y))
    #exit(0)
    
    #torch.set_printoptions(profile = 'full')

    tokenizer = BertTokenizer()
    bert = BertModel.from_pretrained('bert-base-multilingual-cased').to(device)
    fake_bert = Bert(BertConfig()).to(device)

    bmp.load(fake_bert, '/home/hx/lyq/workshop/save_model')
    fake_bert = fake_bert.to(device)
    fake_bert.eval()

    logits = tokenizer([
        "这怎么这么菜啊",
        "才2.9万啊嗯，我还以为9.2万呢",
        ], 
        return_tensors='pt', 
        padding=True
    ).to(device)

    x = bert(**logits).last_hidden_state
    print(x)
    pl()
    fx = fake_bert(**logits).last_hidden_state
    print(fx)

    print(torch.abs(fx - x).max() / torch.abs(x).max())
    print((torch.abs(fx - x) / torch.abs(x)).max())

#    bmp.save(fake_bert, '/home/hx/lyq/workshop/save_model')

if __name__ == "__main__":
    main()
