import time
import random
import tokenizer
import torch
import bmtrain as bmt
import numpy as np
import os
import csv
from bmtrain import nccl
from bmtrain.global_var import config

import model_center as mc
from mc import get_args
from mc.model import CPM1
from mc.tokenizer import CPM1Tokenizer
from mc.data import CPM1_Dataset, DistributedMMapIndexedDataset, MMapIndexedDataset


def get_tokenizer(args):
    tokenizer = CPM1Tokenizer.from_pretrained(args.model_config)
    return tokenizer

def get_model(args):
    model = CPM1.from_pretrained(args.model_config)
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

def make_input(lef_tokens, rig_tokens, spans, max_length):
    input = lef_tokens + [0 for i in range(spans)] + rig_tokens
    length = len(input)

    assert length < max_length # TODO

    input_tokens = torch.zeros((max_length,), dtype=torch.int32)
    input_tokens[:length] = torch.tensor(input).int()

    input_length = torch.tensor(length, dtype=torch.int32)

    context = np.arange(max_length)
    context = (context < len(lef_tokens)) | (context >= len(lef_tokens) + spans)
    context = torch.from_numpy(context).bool()

    input_span = torch.zeros((max_length,), dtype=torch.int32)

    return input_tokens, input_length, context, input_span

class LCQMC_Dataset(torch.utils.data.Dataset):
    def __init__(self, path, rank, world_size, tokenizer, max_length) -> None:
        self.data = []
        with open(path, encoding='utf8') as fin:
            reader = list(csv.reader(fin, delimiter='\t'))
            max_id = (len(reader)-1) // world_size * world_size
            for i, row in enumerate(reader):
                if i==0 or i > max_id: continue
                if i % world_size != rank: continue

                text_a, text_b, label = row
                lef_tokens = [1] + tokenizer.encode(f'"{text_a}"与"{text_b}"的关系是:')
                rig_tokens = tokenizer.encode("。")

                input_tokens, input_length, context, input_span = make_input(lef_tokens, rig_tokens, 1, max_length)

                index = torch.zeros((max_length,), dtype=torch.int32)
                index[len(lef_tokens) - 1] = 1

                target = torch.tensor(int(label), dtype=torch.long)

                self.data.append([
                    input_tokens,
                    input_length,
                    context,
                    input_span,
                    target,
                    index
                ])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return \
            self.data[idx][0].cuda(), \
            self.data[idx][1].cuda(), \
            self.data[idx][2].cuda(), \
            self.data[idx][3].cuda(), \
            self.data[idx][4].cuda(), \
            self.data[idx][5].cuda(), \


def prepare_dataset(args, tokenizer, base_path, rank, world_size):
    splits = ['train', 'dev', 'test']
    suffix = 'tsv' # TODO
    dataset = {}
    for split in splits:
        dataset[split] = LCQMC_Dataset(f"{base_path}/{split}.{suffix}", rank, world_size, tokenizer, args.max_length)
    for split in splits:
        dataset[split] = torch.utils.data.DataLoader(dataset[split], batch_size=args.batch_size, shuffle=(split=='train'))
    verbalizer = torch.LongTensor([15682, 16357]).cuda() # 有关，无关 # TODO
    return dataset, verbalizer


def finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset, verbalizer):
    loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)

    for epoch in range(5):
        model.train()
        for it, (input_tokens, input_length, input_context, input_span, targets, index) in enumerate(dataset['train']):
            # bmt.print_rank(input_tokens[0])

            optimizer.zero_grad()

            logits = model(input_tokens, input_length, input_context, input_span)
            # bmt.print_rank(logits[0])
            logits = logits.index_select(dim=-1, index=verbalizer)
            logits = logits[torch.where(index==1)]

            loss = loss_func(logits, targets)
            global_loss = bmt.sum_loss(loss).item()

            loss = optimizer.loss_scale(loss)
            loss.backward()
            grad_norm = bmt.clip_grad_norm(optimizer.param_groups, args.clip_grad, scale = optimizer.scale / config['world_size'], norm_type = 2)

            bmt.optim_step(optimizer, lr_scheduler)

            bmt.print_rank(
                "train | epoch {:3d} | Iter: {:6d}/{:6d} | loss: {:.4f} | lr: {:.4e}, scale: {:10.4f} | grad_norm: {:.4f} |".format(
                    epoch,
                    it,
                    len(dataset["train"]),
                    global_loss,
                    lr_scheduler.current_lr,
                    int(optimizer.scale),
                    grad_norm
                )
            )
            # if it % args.inspect_iters == 0: bmt.print_inspect(model, "*")
            if args.save != None and it % args.save_iters == 0:
                bmt.save(model, os.path.join(args.save, args.save_name+("-%d.pt" % it)))

        model.eval()
        with torch.no_grad():
            acc = 0
            total = 0
            for it, (input_tokens, input_length, input_context, input_span, targets, index) in enumerate(dataset['dev']):
                logits = model(input_tokens, input_length, input_context, input_span)
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
            for it, (input_tokens, input_length, input_context, input_span, targets, index) in enumerate(dataset['test']):
                logits = model(input_tokens, input_length, input_context, input_span)
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
        "/mnt/sfs_turbo/hx/ModelCenter/down_data/paraphrase/LCQMC",
        bmt.rank(), bmt.world_size(),
    )
    finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset, verbalizer)

if __name__ == "__main__":
    main()
