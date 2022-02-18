#coding:utf-8

import time
import random
import torch
import bmtrain as bmp
import numpy as np
import os

from model import CPM1Config, CPM1
from tokenizer import CPM1Tokenizer
from data import CPM1_Dataset, DistributedMMapIndexedDataset, MMapIndexedDataset

from arguments import get_args
from generation import generate_no_beam

def get_tokenizer(args):
    tokenizer = CPM1Tokenizer(args.vocab_file, space_token = '</_>', line_token = '</n>',)
    return tokenizer

def get_model(args, vocab_size):
    config = CPM1Config.from_json_file(args.model_config)
    config.vocab_size = vocab_size
    print ("vocab size:%d"%(vocab_size))
    model = CPM1(config)
    # if args.load != None:
    bmp.load(model, args.load)
    # else:
    #     bmp.init_parameters(model)
    return model

def setup_model(args):
    tokenizer = get_tokenizer(args)
    model = get_model(args, tokenizer.vocab_size)
    bmp.synchronize()
    bmp.print_rank("Model mem\n", torch.cuda.memory_summary())
    bmp.synchronize()
    return tokenizer, model

def initialize():
    # get arguments
    args = get_args()
    # init bmp 
    bmp.init_distributed(seed = args.seed, loss_scale_factor = 2, loss_scale_steps = 1024)
    # init save folder
    if args.save != None:
        os.makedirs(args.save, exist_ok=True)
    return args

def round_up(x, d):
    return (x + d - 1) // d * d

def make_input(lef_tokens, rig_tokens, spans):
    input = lef_tokens + [0 for i in range(spans)] + rig_tokens
    length = len(input)

    rounded_length = round_up(length, 4)

    input_tokens = torch.zeros(1, rounded_length, dtype=torch.int32)
    input_span = torch.zeros(1, rounded_length, dtype=torch.int32)
    
    context = np.arange((rounded_length))
    context = (context < len(lef_tokens)) | (context >= len(lef_tokens) + spans)
    context = torch.from_numpy(context).view(1, -1).bool()

    input_length = torch.zeros(1, dtype=torch.int32)
    input_tokens[0, :length] = torch.tensor(input).int()
    input_length[0] = length

    print (input_tokens)
    print (input_length)
    print (input_span)
    print (context)

    return input_tokens.cuda(), input_length.cuda(), input_span.cuda(), context.cuda()

def tokenize(tokenizer, sentence):
    return [1] + tokenizer.encode(sentence)

def generate(lef_sentence, rig_sentence, spans, tokenizer, model, topk=1):

    lef_tokens = tokenizer.encode(lef_sentence)
    rig_tokens = tokenizer.encode(rig_sentence)
    lef_tokens = [1] + lef_tokens

    input_tokens, input_length, input_span, context = make_input(lef_tokens, rig_tokens, spans)
    yield lef_sentence

    with torch.inference_mode():
        for i in range(len(lef_tokens) - 1, len(lef_tokens) + spans - 2):
            logits = model(input_tokens, input_length, context, input_span)
            # assert input_tokens[0][i+1] == 0
            # assert context[0][i] == True and  context[0][i+1] == False
            logits = logits[0, i, :].view(-1)
            # print (torch.topk(logits, topk, sorted=True))
            vocab_idx = logits.argmax().cpu().item()
            input_tokens[0][i + 1] = vocab_idx
            # context[0][i+1] = True
            yield tokenizer.decode([vocab_idx])
    yield rig_sentence


# def get_ppl(sentA : str, results : List[str], tokenizer : T5Tokenizer, model : T5):
#     with torch.inference_mode():
#         enc_tensor, enc_len = make_input( tokenize(tokenizer, sentA) )
#         dec_input = []
#         dec_target = []
#         for i, r in enumerate(results):
#             tokens = tokenizer.encode(r)
#             span_idx = tokenizer.get_span(i)
#             dec_input.append( span_idx )
#             dec_target.append( span_idx )
#             dec_input.extend( tokens )
#             dec_target.extend( tokens )
        
#         dec_target.append( tokenizer.eod_id )
#         dec_target = dec_target[1:]

        
#         dec_tensor, dec_len = make_input(dec_input)
#         while len(dec_target) < dec_tensor.size(1):
#             dec_target.append(-100)
#         target_tensor = torch.tensor([dec_target]).long().cuda()
    
#         logits = model(enc_tensor, enc_len, dec_tensor, dec_len)
#         loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)
#         batch, seq_len, vocab_out_size = logits.size()
#         loss = loss_func(logits.view(batch * seq_len, vocab_out_size), target_tensor.view(batch * seq_len))
#         loss = loss.cpu().item()
#         print(enc_tensor.size(), dec_tensor.size())
#         print("Loss: %lf" % loss)
#         print("PPL: %lf" % math.exp(loss))

def main():
    args = initialize()
    tokenizer, model = setup_model(args)

    # input_sentence = 
    # lef_sentence = "据北京环球度假区介绍,园区开园前的各项准备工作已"
    # rig_sentence = "已经盛大开业"
    # spans = 16

    lef_sentence = "苏联的一次大会上,主持人突然说到:"
    rig_sentence = "主持人慌忙说:那请您赶快坐到主席台上来"
    spans = 32

    # lef_sentence = "二甲双胍对于冠状动脉粥样硬化的效果不明"
    # rig_sentence = "该结果有待进一步观察"
    # spans = 128
    # lef_sentence = "兔子有"
    # rig_sentence = "条腿"
    # spans = 1


    # dataset = CPM1_Dataset(
    #     DistributedMMapIndexedDataset(f"{args.base_path}/new_data/", "cpm1_lm_document_context", 0, 32),
    #     DistributedMMapIndexedDataset(f"{args.base_path}/new_data/", "cpm1_lm_document_context", 0, 32),
    # )

    fout = open(f"{args.base_path}/text.out", "w", encoding="utf-8")
    # for st in range(0, 52):
    #     data = dataset[st]
    #     if data is None:
    #         continue
    #     aa = tokenizer.decode(data["ctx"].cpu().numpy())
    #     fout.write(aa+"\n")



    
    # for j in range(1):
    #     dataset = CPM1_Dataset(
    #         DistributedMMapIndexedDataset(f"{args.base_path}/new_data/", "cpm1_lm_document_context", j, 32),
    #         DistributedMMapIndexedDataset(f"{args.base_path}/new_data/", "cpm1_lm_document_target", j, 32)
    #     )
    #     # 10501*args.batch_size
    #     for st in range(13166*52,13167*52):
    #         data = dataset[st]
    #         if data is None:
    #             continue
    #         input_length = torch.LongTensor([data["len_ctx"]]).int().cuda()
    #         input_idx = torch.stack([data["ctx"]]).int().cuda()
            
    #         logits = model(input_idx, input_length)
    #         print ("============")
    #         print(logits[0,0,:].max(-1))
    #         print(logits[0,0,:].mean(-1))
    #         print(logits[0,0,data["ctx"][1]])

    #         print(logits[0,1,:].max(-1))
    #         print(logits[0,1,:].mean(-1))
    #         print(logits[0,10,:].max(-1))
    #         print(logits[0,10,:].mean(-1))
    #         print ("------------")


    #         # aa = it["ctx"].cpu().numpy()
    #         # bb = it["tgt"].cpu().numpy()
    #         # res = tokenizer.decode(aa)
    #         # fout.write(res+"\n")        
    #         # res = tokenizer.decode(bb)
    #         # fout.write(res+"\n")
    #         # for hx in range(0, len(aa)-2):
    #         #     if bb[hx] == 4:
    #         #         break
    #         #     assert aa[hx + 1] == bb[hx], print(aa[hx+1],bb[hx], hx)

    for it in generate_no_beam(model, tokenizer, lef_sentence, rig_sentence, spans):
        fout.write(it)
        fout.flush()
    fout.close()

if __name__ == "__main__":
    main()
