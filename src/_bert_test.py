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

    #torch.set_printoptions(profile = 'full')

    tokenizer = BertTokenizer()
    bert = BertModel.from_pretrained('bert-base-multilingual-cased').to(device)
    fake_bert = Bert(BertConfig()).to(device)
    
#    for k in fake_bert.state_dict():
#        print(k)
#    exit(0)

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
    pl()

    print(torch.abs(fx - x).max() / torch.abs(x).max())
    print((torch.abs(fx - x)).max())

#    bmp.save(fake_bert, '/home/hx/lyq/workshop/save_model')

if __name__ == "__main__":
    main()
