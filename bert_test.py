#coding:utf-8

import torch
import bmtrain as bmt

from tokenizer import BertTokenizer
from model import BertConfig, Bert

from transformers import BertModel
from arguments import get_args

def get_tokenizer(args):
    return BertTokenizer()

def get_model(args):
    config = BertConfig.from_pretrained(args.model_config)
    model = Bert(config)
    bmt.load(model, args.load)#'/home/hx/lyq/workshop/save_model')
    return model

def main():
    args = get_args()
    bmt.init_distributed()

    device = 'cuda:0'

    tokenizer = get_tokenizer(args)
    fake_bert = get_model(args).to(device)
    bert = BertModel.from_pretrained('bert-base-multilingual-cased').to(device).half()
    
    bert.eval()
    fake_bert.eval()

    logits = tokenizer([
        "这怎么这么菜啊",
        "才2.9万啊嗯，我还以为9.2万呢",
        ], 
        return_tensors='pt', 
        padding=True
    ).to(device)

    batch_size, seq_length = logits['input_ids'].size()
    mask = logits['attention_mask']
    mask = mask.view(batch_size, seq_length, 1).repeat(1, 1, 768)

    x = bert(**logits)['last_hidden_state']
    fx = fake_bert(**logits)['last_hidden_state']

    fx = fx.transpose(1,2)

    print(fx,x,sep='\n')
    print(torch.abs(fx - x).max() / torch.abs(x).max())
    print((torch.abs(fx - x)).max())
    print(torch.abs(fx - x).sum() / x.numel())
    print((torch.abs(fx - x) > 0.1).sum())

if __name__ == "__main__":
    main()
