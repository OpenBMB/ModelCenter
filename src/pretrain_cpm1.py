import time
import random
import torch
import bmpretrain as bmp
from bmpretrain import nccl
from bmpretrain.global_var import config
import numpy as np
import os
from model import CPM1Config, CPM1
from tokenizer import CPM1Tokenizer
from data import CPM1_Dataset, DistributedMMapIndexedDataset, MMapIndexedDataset, CPM1_Dataset_Merge
from arguments import get_args
import distutils.version
from torch.utils.tensorboard import SummaryWriter

def get_tokenizer(args):
    tokenizer = CPM1Tokenizer(args.vocab_file, space_token = '</_>', line_token = '</n>',)
    return tokenizer

def get_model(args, vocab_size):
    config = CPM1Config.from_json_file(args.model_config)
    config.vocab_size = vocab_size
    print ("vocab size:%d"%(vocab_size))
    model = CPM1(config)
    if args.load != None:
        bmp.load(model, args.load)
    else:
        bmp.init_parameters(model)
    return model

def get_optimizer(args, model):
    optimizer = bmp.optim.AdamOffloadOptimizer(model.parameters(), 
                                               weight_decay=args.weight_decay, 
                                               scale=args.loss_scale)
    return optimizer

def get_learning_rate_scheduler(args, optimizer):
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters * args.epochs
    lr_scheduler = bmp.lr_scheduler.Noam(optimizer, 
                                         start_lr = args.lr,
                                         warmup_iter = args.warmup_iters, 
                                         end_iter = args.lr_decay_iters,
                                         num_iter = args.start_step)
    return lr_scheduler

def setup_model_and_optimizer(args):
    # get the tokenizer
    tokenizer = get_tokenizer(args)
    # get the model
    model = get_model(args, tokenizer.vocab_size)
    bmp.synchronize()
    # get the optimizer and lr_scheduler
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    bmp.synchronize()
    # get the memory usage
    bmp.print_rank("Model mem\n", torch.cuda.memory_summary())
    bmp.synchronize()
    return tokenizer, model, optimizer, lr_scheduler

def initialize():
    # get arguments
    args = get_args()
    # init bmp 
    bmp.init_distributed(seed = args.seed, loss_scale_factor = 2, loss_scale_steps = 1024)
    # init save folder
    if args.save != None:
        os.makedirs(args.save, exist_ok=True)
    return args

def batch_iter(args, dataset, start_step = 0):
    st = 0
    ctx = []
    tgt = []
    context = []
    span = []
    hx = 0
    while True:
        ctx_data, tgt_data, _len, context_data = dataset[st]
        st += 1
        if ctx_data is None:
            continue
        assert _len <= args.max_length

        ctx_data = ctx_data.astype("int64")
        tgt_data = tgt_data.astype("int64")

        for index in range(len(ctx)):
            if span[index][-1] + _len < args.max_length:
                ctx[index][span[index][-1]:span[index][-1] + _len] = torch.from_numpy(ctx_data)[:_len].long()
                tgt[index][span[index][-1]:span[index][-1] + _len]= torch.from_numpy(tgt_data)[:_len].long()
                context[index][span[index][-1]:span[index][-1] + _len] = torch.from_numpy(context_data)[:_len].bool()
                span[index].append(span[index][-1] + _len)
                break
        else:
            _ctx = torch.zeros((args.max_length,), dtype=torch.long)
            _ctx[:_len] = torch.from_numpy(ctx_data)[:_len].long()
            _tgt = torch.full((args.max_length,), -100, dtype=torch.long)
            _tgt[:_len] = torch.from_numpy(tgt_data)[:_len].long()
            _context = torch.full((args.max_length,), False, dtype=torch.bool)
            _context[:_len] = torch.from_numpy(context_data)[:_len].bool()

            ctx.append(_ctx)
            tgt.append(_tgt)
            context.append(_context)
            span.append([_len])

        if len(ctx) > args.batch_size:
            if hx >= start_step:

                _span = torch.zeros((args.batch_size, args.max_length + 1), dtype=torch.long)
                for bindex in range(args.batch_size):
                    for sindex in span[bindex]:
                        _span[bindex][sindex] = 1

                yield {
                    "ctx": torch.stack(ctx[:args.batch_size]),
                    "tgt": torch.stack(tgt[:args.batch_size]),
                    "context": torch.stack(context[:args.batch_size]),
                    "span": torch.cumsum(_span, dim=-1)[:,:-1],
                    "len_ctx": torch.LongTensor([it[-1] for it in span[:args.batch_size]]),
                }

            hx += 1
            ctx = ctx[args.batch_size:]
            tgt = tgt[args.batch_size:]
            context = context[args.batch_size:]
            span = span[args.batch_size:]

def print_inspect(model, name):
    bmp.print_rank(
        bmp.inspect.format_summary(
            bmp.inspect.inspect_model(model, name)
        )
    )

def clip_grad_norm(param_groups, max_norm, scale, norm_type=2, eps=1e-6):

    parameters = [p for group in param_groups for p in group['params'] if p.grad is not None]

    if norm_type == 'inf':
        total_norm_cuda = max(p.grad.data.abs().max() for p in parameters).detach()
        nccl.allReduce(total_norm_cuda.storage(), total_norm_cuda.storage(), "max", config["comm"])
        total_norm = total_norm_cuda
    else:
        norm_type = float(norm_type)
        total_norm_cuda = torch.cuda.FloatTensor([0])
        for p in parameters:
            param_norm = p.grad.data.float().norm(norm_type)
            total_norm_cuda += param_norm ** norm_type
        nccl.allReduce(total_norm_cuda.storage(), total_norm_cuda.storage(), "sum", config["comm"])
        total_norm = total_norm_cuda[0] ** (1. / norm_type)

    # total_norm = total_norm / scale
    # clip_coef = float(max_norm) / (total_norm + eps)
    clip_coef = float(max_norm * scale) / (total_norm + eps)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    return total_norm / scale

def pretrain(args, tokenizer, model, optimizer, lr_scheduler, dataset):
    average_time = 0
    average_time_shift = 0.9
    loss_func = bmp.loss.FusedCrossEntropy(ignore_index=-100)
    loss_func_tmp = torch.nn.CrossEntropyLoss(ignore_index=-100, reduce = False)


    if bmp.rank() == 0:
        writer = SummaryWriter("runs/cpm-1")

    start_step = args.start_step

    for iteration, data in enumerate(batch_iter(args, dataset, start_step)):
        iteration = iteration + start_step

        st = time.time()
        optimizer.zero_grad()

        assert len(data["ctx"]) == args.batch_size

        input_idx = data["ctx"].int().cuda()
        input_length = data["len_ctx"].int().cuda()
        input_context = data["context"].bool().cuda()
        input_span = data["span"].int().cuda()
        targets = data["tgt"].long().cuda()

        logits = model(input_idx, input_length, input_context, input_span)
        # loss_1 = loss_func_tmp(logits.view(-1, logits.size(-1)), targets.view(-1))
        # print (loss_1.max(), "==========", (loss_1>10).sum(), loss_1.mean())
        
        loss = loss_func(logits.view(-1, logits.size(-1)), targets.view(-1))
        global_loss = bmp.sum_loss(loss).item()

        loss = optimizer.loss_scale(loss)
        loss.backward()
        grad_norm = clip_grad_norm(optimizer.param_groups, args.clip_grad, scale = optimizer.scale / config['world_size'], norm_type = 2)

        bmp.optim_step(optimizer, lr_scheduler)

        iteration_time = time.time() - st
        average_time = average_time * average_time_shift + (1 - average_time_shift) * iteration_time

        bmp.print_rank(
                "| Iter: {:6d} | loss: {:.4f} | lr: {:.4e}, scale: {:10.4f} | time: {:.4f} | token/max: {:.4f} | mask/max: {:.4f} | grad_norm: {:.4f}".format(
                    iteration,
                    global_loss,
                    lr_scheduler.current_lr,
                    int(optimizer.scale),
                    average_time / (1 - pow(average_time_shift, iteration + 1)),
                    input_length.float().mean()/args.max_length,
                    (targets>=0).sum(-1).float().mean()/args.max_length,
                    grad_norm
                )
            )

        if iteration % args.inspect_iters == 0:
            print_inspect(model, "*")
        if bmp.rank() == 0:
            writer.add_scalar("Loss/train", global_loss, iteration + start_step)
        if args.save != None and iteration % args.save_iters == 0:
            bmp.save(model, os.path.join(args.save, args.save_name+("-%d.pt" % iteration)))

    bmp.save(model, os.path.join(args.save, args.save_name+".pt"))

def main():
    args = initialize()
    tokenizer, model, optimizer, lr_scheduler = setup_model_and_optimizer(args)
    dataset = CPM1_Dataset_Merge(
        DistributedMMapIndexedDataset("/mnt/sfs_turbo/hx/cpm3-pretrain/new_data/", "cpm1_lm_document_context", bmp.rank(), bmp.world_size()), 
        args.max_length
    )
    pretrain(args, tokenizer, model, optimizer, lr_scheduler, dataset)

if __name__ == "__main__":
    main()
