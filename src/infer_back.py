#coding:utf-8

import time
import random
import torch
import bmtrain as bmt
import numpy as np
import os

from model import CPM1Config, CPM1
from tokenizer import CPM1Tokenizer
from data import CPM1_Dataset, DistributedMMapIndexedDataset, MMapIndexedDataset

from arguments import get_args

def get_tokenizer(args):
    tokenizer = CPM1Tokenizer(args.vocab_file, space_token = '</_>', line_token = '</n>',)
    return tokenizer

def get_model(args, vocab_size):
    config = CPM1Config.from_json_file(args.model_config)
    config.vocab_size = vocab_size
    print ("vocab size:%d"%(vocab_size))
    model = CPM1(config)
    if args.load != None:
        bmt.load(model, args.load)
    else:
        bmt.init_parameters(model)
    return model

def setup_model(args):
    tokenizer = get_tokenizer(args)
    model = get_model(args, tokenizer.vocab_size)
    bmt.synchronize()
    bmt.print_rank("Model mem\n", torch.cuda.memory_summary())
    bmt.synchronize()
    return tokenizer, model

def initialize():
    # get arguments
    args = get_args()
    # init bmt 
    bmt.init_distributed(seed = args.seed, loss_scale_factor = 2, loss_scale_steps = 1024)
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

    return input_tokens.cuda(), input_length.cuda(), input_span.cuda(), context.cuda()

def tokenize(tokenizer, sentence):
    return [1] + tokenizer.encode(sentence)

def generate(lef_sentence, rig_sentence, spans, tokenizer, model, topk=1):

    def takeSecond(elem):
        return elem[1]

    input_tokens, input_length, input_span, context = make_input(lef_sentence, rig_sentence, spans)
    yield lef_sentence

    with torch.inference_mode():
        for i in range(spans):
            new_ipt = []

            for ipt, logit in dec_ipt:
                dec_tensor, dec_len, dec_span, dec_context = make_input(ipt)

                logits = model(dec_tensor, dec_len, dec_context, dec_span)

                logits = logits[0, len(ipt) - 1, :].view(-1)
                score_topk, index_topk = torch.topk(logits, topk, sorted=True)

                score_topk = score_topk.cpu()
                index_topk = index_topk.cpu()

                for score, index in zip(score_topk, index_topk):
                    new_ipt.append((ipt+[index.item()], logit+score.item()))

            new_ipt.sort(key=takeSecond, reverse=True)
            print ([i[1] for i in new_ipt])
            print ("=======================")
            dec_ipt = new_ipt[:topk]

        dec_ipt.sort(key=takeSecond, reverse=True)
        for i in dec_ipt[0][0]:
            yield tokenizer.decode([i])

    yield rig_sentence

        # yield "\n"
        # for i in dec_ipt[1][0]:
        #     yield tokenizer.decode([i])

        # yield "\n"
        # for i in dec_ipt[2][0]:
        #     yield tokenizer.decode([i])

        # yield "\n"
        # for i in dec_ipt[3][0]:
        #     yield tokenizer.decode([i])

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
    # input_sentence = "北京环球度假区相关负责人表示,“我们将继续加大对北京环球度假区的支持力度"
    lef_sentence = "今天(7月1日)上午,在南宁市莱茵鹭湖小区,"
    rig_sentence = "事件导致多人不同程度受伤｡"
    spans = 10

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



    for it in generate(lef_sentence, rig_sentence, spans, tokenizer, model):
        fout.write(it)
        fout.flush()
    fout.close()

if __name__ == "__main__":
    main()
