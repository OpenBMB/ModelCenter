import time
import random

from sklearn.metrics import accuracy_score, f1_score
import torch
import bmtrain as bmp
from bmtrain import nccl
from bmtrain.global_var import config
import numpy as np
import os
import csv
from dataset.gpt2dataset import DATASET

from model import GPT2Config, GPT2
from transformers import GPT2Tokenizer

from arguments import get_args

def get_tokenizer(args):
    tokenizer = GPT2Tokenizer.from_pretrained(f"{args.base_path}/vocab/gpt2")
    return tokenizer

def get_model(args, vocab_size):
    config = GPT2Config.from_json_file(args.model_config)
    config.vocab_size = vocab_size
    print("vocab size:%d"%(vocab_size))
    from transformers import GPT2LMHeadModel
    model = GPT2LMHeadModel.from_pretrained(f"{args.base_path}/vocab/gpt2").cuda()
    return model

def get_optimizer(args, model):
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay)
    return optimizer

def get_learning_rate_scheduler(args, optimizer):
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters * args.epochs
    lr_scheduler = None
    return lr_scheduler

def setup_model_and_optimizer(args):
    # get the tokenizer
    tokenizer = get_tokenizer(args)
    # get the model
    model = get_model(args, 50258) # tokenizer.vocab_size is 50257 which is odd number
    # get the optimizer and lr_scheduler
    optimizer = get_optimizer(args, model)
    lr_scheduler = get_learning_rate_scheduler(args, optimizer)
    # get the memory usage
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


def prepare_dataset(args, tokenizer, base_path, dataset_name, rank, world_size):
    splits = ['train', 'dev', 'test']
    dataset = {}
    for split in splits:
        dataset[split] = DATASET[dataset_name](base_path, split, rank, world_size, tokenizer, args.max_decoder_length)
    verbalizer = torch.LongTensor(DATASET[dataset_name].get_verbalizer(tokenizer)).cuda()
    return dataset, verbalizer


def metric(gts, pds, qids):
# Copyright 2020 The HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
    f1_a = float(f1_score(gts, pds))
    ss = {}
    for gt, pd, qid in zip(gts, pds, qids):
        if qid not in ss: ss[qid] = []
        ss[qid].append((gt, pd))
    f1s, ems = [], []
    for qid, gt_pd_s in ss.items():
        gts, pds = zip(*gt_pd_s)
        f1s.append(f1_score(gts, pds, average="macro"))
        ems.append(int(sum([gt == pd for gt, pd in gt_pd_s])==len(gt_pd_s)))
    f1_m = float((sum(f1s) / len(f1s)))
    em = sum(ems) / len(ems)
    return f1_a, f1_m, em

def finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset, verbalizer):
    loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100)

    # bmp.print_inspect(model, '*')

    for epoch in range(20):
        torch.manual_seed(233)
        split_length = int(len(dataset["train"])*0.9)
        trainset, devset = torch.utils.data.random_split(dataset["train"], [split_length, len(dataset["train"])-split_length])
        testset = dataset["dev"]
        dataloader = {
            "train": torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True),
            "dev": torch.utils.data.DataLoader(devset, batch_size=args.batch_size, shuffle=False),
            "test": torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False),
        }
        print(len(dataloader["train"]))
        print(len(dataloader["test"]))

        model.train()
        for it, data in enumerate(dataloader['train']):
            input_ids = data["input_ids"]
            input_length = data["input_length"]
            labels = data["labels"]
            targets = data["targets"]
            index = data["index"]

            optimizer.zero_grad()

            mask_1d = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :].repeat(input_ids.shape[0], 1) < input_length[:, None]
            outputs = model(input_ids, attention_mask = mask_1d, output_hidden_states=True, output_attentions=True)
            logits = outputs.logits

            loss = loss_func(logits.view(-1, logits.shape[-1]), targets.view(-1))

            logits = logits.index_select(dim=-1, index=verbalizer)
            logits = logits[torch.where(index==1)]
            loss = loss + loss_func(logits, labels)

            global_loss = loss.item()

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad, norm_type = 2)

            optimizer.step()

            bmp.print_rank(
                "train | epoch {:3d} | Iter: {:6d}/{:6d} | loss: {:.4f} | lr: {:.4e}, scale: {:10.4f} | grad_norm: {:.4f} |".format(
                    epoch,
                    it,
                    len(dataloader["train"]),
                    global_loss,
                    0, #lr_scheduler.current_lr,
                    0, #int(optimizer.scale),
                    grad_norm,
                )
            )
            # if it % args.inspect_iters == 0: bmp.print_inspect(model, "*")
            # if args.save != None and it % args.save_iters == 0:
            #     bmp.save(model, os.path.join(args.save, args.save_name+("-%d.pt" % it)))

        model.eval()
        with torch.no_grad():
            for split in ['dev', 'test']:
                pd = []
                gt = []
                # print(len(dataloader[split]))
                for it, data in enumerate(dataloader[split]):
                    input_ids = data["input_ids"]
                    input_length = data["input_length"]
                    labels = data["labels"]
                    index = data["index"]

                    mask_1d = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :].repeat(input_ids.shape[0], 1) < input_length[:, None]
                    outputs = model(input_ids, attention_mask = mask_1d, output_hidden_states=True, output_attentions=True)
                    logits = outputs.logits
                    logits = logits.index_select(dim=-1, index=verbalizer)
                    logits = logits[torch.where(index==1)]
                    logits = logits.argmax(dim=-1)
                
                    pd.extend(logits.cpu().tolist())
                    gt.extend(labels.cpu().tolist())

                    bmp.print_rank(
                        "{} | epoch {:3d} | Iter: {:6d}/{:6d} |".format(
                            split,
                            epoch,
                            it,
                            len(dataloader[split]),
                        )
                    )
                pd_local = torch.tensor(pd).int().cuda()
                gt_local = torch.tensor(gt).int().cuda()
                pd_global = torch.empty((len(pd)*config["world_size"],), dtype=torch.int32).cuda()
                gt_global = torch.empty((len(gt)*config["world_size"],), dtype=torch.int32).cuda()
                bmp.synchronize()
                nccl.allGather(pd_local.storage(), pd_global.storage(), config["comm"])
                nccl.allGather(gt_local.storage(), gt_global.storage(), config["comm"])
                bmp.synchronize()
                pd = pd_global.cpu().tolist()
                gt = gt_global.cpu().tolist()
                bmp.print_rank(pd)
                bmp.print_rank(gt)
                
                bmp.print_rank(f"{split} epoch {epoch}:")
                if args.dataset_name in ["BoolQ", "CB", "COPA", "RTE", "WiC", "WSC"]:
                    acc = accuracy_score(gt, pd)
                    bmp.print_rank(f"accuracy: {acc*100:.2f}")
                if args.dataset_name in ["CB"]:
                    f1 = f1_score(gt, pd, average="macro")
                    bmp.print_rank(f"Average F1: {f1*100:.2f}")
                if args.dataset_name in ["MultiRC", "ReCoRD"]:
                    qids = devset.qids if split == 'dev' else testset.qids
                    f1_a, _, em = metric(gt, pd, qids)
                    # TODO qids also need to allGather
                    bmp.print_rank(f"F1: {f1_a*100:.2f}")
                    bmp.print_rank(f"EM: {em*100:.2f}")


def main():
    args = initialize()
    tokenizer, model, optimizer, lr_scheduler = setup_model_and_optimizer(args)
    dataset, verbalizer = prepare_dataset(
        args,
        tokenizer,
        f"{args.base_path}/down_data/superglue/",
        args.dataset_name,
        0, 1
    )
    finetune(args, tokenizer, model, optimizer, lr_scheduler, dataset, verbalizer)

if __name__ == "__main__":
    main()
