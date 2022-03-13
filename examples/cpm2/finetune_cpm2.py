import time
import random
import tokenizer
import torch
import bmtrain as bmt
from bmtrain import nccl
from bmtrain.global_var import config
import numpy as np
import os
import csv

import model_center as mc
from mc import get_args
from mc.model import CPM2
from mc.tokenizer import CPM2Tokenizer


def get_tokenizer(args):
    tokenizer = CPM2Tokenizer.from_pretrained(args.model_config)
    return tokenizer

def get_model(args):
    model = CPM2.from_pretrained(args.model_config)
    return model

def get_optimizer(args, model):
    optimizer = bmt.optim.AdamOffloadOptimizer(model.parameters(), 
                                               weight_decay=args.weight_decay, 
                                               scale=args.loss_scale)
    return optimizer

def get_learning_rate_scheduler(args, optimizer):
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters * args.epochs
    lr_scheduler = bmt.lr_scheduler.Noam(optimizer, 
                                         start_lr = args.lr,
                                         warmup_iter = args.warmup_iters, 
                                         end_iter = args.lr_decay_iters,
                                         num_iter = args.start_step)
    return lr_scheduler

def setup_model_and_optimizer(args):
    # get the tokenizer
    tokenizer = get_tokenizer(args)
    # get the model
    model = get_model(args)
    bmt.synchronize()
    # get the optimizer and lr_scheduler
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    bmt.synchronize()
    # get the memory usage
    bmt.print_rank("Model mem\n", torch.cuda.memory_summary())
    bmt.synchronize()
    return tokenizer, model, optimizer, lr_scheduler

def initialize():
    # get arguments
    args = get_args()
    # init bmt 
    bmt.init_distributed(seed = args.seed, loss_scale_factor = 2, loss_scale_steps = 1024)
    # init save folder
    if args.save != None:
        os.makedirs(args.save, exist_ok=True)
    return args

def make_input(tokenizer, input, max_encoder_length, max_decoder_length):
    input = input + [tokenizer.get_sentinel_id(0)]
    length = len(input)

    assert length < max_encoder_length # TODO

    input_tokens = torch.zeros((max_encoder_length,), dtype=torch.int32)
    input_tokens[:length] = torch.tensor(input).int()

    input_length = torch.tensor(length, dtype=torch.int32)

    output = [tokenizer.get_sentinel_id(0)]
    length = len(output)
    output_tokens = torch.zeros((max_decoder_length,), dtype=torch.int32)
    output_tokens[:length] = torch.tensor(output).int()
    output_length = torch.tensor(length, dtype=torch.int32)

    index = torch.zeros((max_decoder_length,), dtype=torch.int32)
    index[length - 1] = 1

    return input_tokens, input_length, output_tokens, output_length, index

class LCQMC_Dataset(torch.utils.data.Dataset):
    def __init__(self, path, split, rank, world_size, tokenizer, max_encoder_length, max_decoder_length) -> None:
        path = f"{path}/{split}.tsv"
        self.data = []
        with open(path, encoding='utf8') as fin:
            reader = list(csv.reader(fin, delimiter='\t'))[1:]
            max_id = (len(reader)) // world_size * world_size
            for i, row in enumerate(reader):
                if i > max_id: continue
                if i % world_size != rank: continue

                text_a, text_b, label = row
                enc_input = tokenizer.encode(f'“{text_a}”与“{text_b}”是否有关？')

                enc_tokens, enc_length, dec_tokens, dec_length, index = make_input(tokenizer, enc_input, max_encoder_length, max_decoder_length)

                target = torch.tensor(int(label), dtype=torch.long)

                self.data.append({
                    "enc_input": enc_tokens.cuda(),
                    "enc_length": enc_length.cuda(),
                    "dec_input": dec_tokens.cuda(),
                    "dec_length": dec_length.cuda(),
                    "targets": target.cuda(),
                    "index": index.cuda(),
                })

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


def prepare_dataset(args, tokenizer, base_path, rank, world_size):
    splits = ['train', 'dev', 'test']
    dataset = {}
    for split in splits:
        dataset[split] = LCQMC_Dataset(base_path, split, rank, world_size, tokenizer, args.max_encoder_length, args.max_decoder_length)
    for split in splits:
        dataset[split] = torch.utils.data.DataLoader(dataset[split], batch_size=args.batch_size, shuffle=(split=='train'))
    verbalizer = torch.LongTensor([1744, 24]).cuda()
    return dataset, verbalizer


def finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset, verbalizer):
    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)

    for epoch in range(5):
        model.train()
        for it, data in enumerate(dataset['train']):
            enc_input = data["enc_input"]
            enc_length = data["enc_length"]
            dec_input = data["dec_input"]
            dec_length = data["dec_length"]
            targets = data["targets"]
            index = data["index"]

            optimizer.zero_grad()

            logits = model(enc_input, enc_length, dec_input, dec_length)
            logits = logits.index_select(dim=-1, index=verbalizer)
            logits = logits[torch.where(index==1)]

            loss = loss_func(logits, targets)
            global_loss = bmt.sum_loss(loss).item()

            loss = optimizer.loss_scale(loss)
            loss.backward()
            grad_norm = bmt.clip_grad_norm(optimizer.param_groups, args.clip_grad, scale = optimizer.scale / bmt.world_size(), norm_type = 2)

            bmt.optim_step(optimizer, lr_scheduler)

            bmt.print_rank(
                "train | epoch {:3d} | Iter: {:6d}/{:6d} | loss: {:.4f} | lr: {:.4e}, scale: {:10.4f} | grad_norm: {:.4f} |".format(
                    epoch,
                    it,
                    len(dataset["train"]),
                    global_loss,
                    lr_scheduler.current_lr,
                    int(optimizer.scale),
                    grad_norm,
                )
            )
            # if it % args.inspect_iters == 0: bmt.print_inspect(model, "*")
            if args.save != None and it % args.save_iters == 0:
                bmt.save(model, os.path.join(args.save, args.save_name+("-%d.pt" % it)))

        model.eval()
        with torch.no_grad():
            acc = 0
            total = 0
            for it, data in enumerate(dataset['dev']):
                enc_input = data["enc_input"]
                enc_length = data["enc_length"]
                dec_input = data["dec_input"]
                dec_length = data["dec_length"]
                targets = data["targets"]
                index = data["index"]

                logits = model(enc_input, enc_length, dec_input, dec_length)
                logits = logits.index_select(dim=-1, index=verbalizer)
                logits = logits[torch.where(index==1)]
                logits = logits.argmax(dim=-1)
            
                acc += torch.sum(logits == targets).item()
                total += logits.shape[0]
                bmt.print_rank(
                    "dev | epoch {:3d} | Iter: {:6d}/{:6d} | acc: {:6d} | total: {:6d} |".format(
                        epoch,
                        it,
                        len(dataset["dev"]),
                        acc,
                        total,
                    )
                )
            acc = torch.tensor(acc / total).cuda()
            acc = bmt.sum_loss(acc).cpu().item()
            bmt.print_rank(f"dev epoch {epoch}: accuracy: {acc}")

        with torch.no_grad():
            acc = 0
            total = 0
            for it, data in enumerate(dataset['test']):
                enc_input = data["enc_input"]
                enc_length = data["enc_length"]
                dec_input = data["dec_input"]
                dec_length = data["dec_length"]
                targets = data["targets"]
                index = data["index"]

                logits = model(enc_input, enc_length, dec_input, dec_length)
                logits = logits.index_select(dim=-1, index=verbalizer)
                logits = logits[torch.where(index==1)]
                logits = logits.argmax(dim=-1)

                acc += torch.sum(logits == targets).item()
                total += logits.shape[0]
                bmt.print_rank(
                    "test | epoch {:3d} | Iter: {:6d}/{:6d} | acc: {:6d} | total: {:6d} |".format(
                        epoch,
                        it,
                        len(dataset["test"]),
                        acc,
                        total,
                    )
                )
            acc = torch.tensor(acc / total).cuda()
            acc = bmt.sum_loss(acc).cpu().item()
            bmt.print_rank(f"test epoch {epoch}: accuracy: {acc}")

def main():
    args = initialize()
    tokenizer, model, optimizer, lr_scheduler = setup_model_and_optimizer(args)
    dataset, verbalizer = prepare_dataset(
        args,
        tokenizer,
        f"{args.base_path}/down_data/paraphrase/LCQMC",
        bmt.rank(), bmt.world_size(),
    )
    finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset, verbalizer)

if __name__ == "__main__":
    main()
