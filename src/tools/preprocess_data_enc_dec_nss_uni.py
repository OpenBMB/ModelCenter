# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Processing data for pretraining."""

import argparse
import json
import multiprocessing
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time
import math
import torch
import torch.nn.functional as F
from tqdm import tqdm
import random
import numpy as np

from tokenization_enc_dec import EncDecTokenizer
from data import indexed_dataset

# from ray.util.multiprocessing.pool import Pool

random.seed(233)
np.random.seed(233)
g = torch.manual_seed(233)
torch.cuda.manual_seed_all(233)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = EncDecTokenizer(os.path.join(self.args.tokenizer_path, 'vocab.txt'))

    def encode(self, all_lines):

        stat_map = [
            (-1, -1), # 0,
            (-1, -1), # 1,
            (400, 50, [1, 1, [0, 0, 0, 0]]), # 2,
            (400, 50, [1, 1, [1, 0, 0, 0]]), # 3,
            (400, 50, [1, 1, [1, 1, 0, 0]]), # 4,
            (400, 40, [1, 1, [1, 1, 1, 0]]), # 5,
            (300, 40, [1, 1, [1, 1, 1, 1]]), # 6,
            (250, 30, [1, 2, [1, 1, 1, 1]]), # 7,
            (200, 30, [1, 2, [2, 1, 1, 1]]), # 8,
            (200, 30, [1, 2, [2, 2, 1, 1]]), # 9,
            (150, 20, [1, 2, [2, 2, 2, 1]]), # 10,
            (150, 20, [1, 3, [2, 2, 2, 1]]), # 11,
            (150, 20, [1, 3, [2, 2, 2, 2]]), # 12,
            (150, 20, [1, 3, [3, 2, 2, 2]]), # 13,
            (150, 20, [1, 3, [3, 3, 2, 2]]), # 14,
            (150, 20, [1, 3, [3, 3, 3, 2]]), # 15,
            (150, 20, [1, 3, [3, 3, 3, 3]]), # 16,
        ]

        # end with <eod>
        all_doc_ids = []

        n = int(random.random() / (1 / 15)) + 2

        for i, line in enumerate(all_lines):
            if len(line) > 5000000:
                return None, None, 0

            # data
            data = line.strip()
            data = data.split("<n>")

            if min([len(x) for x in data]) < 10:
                return None, None, 0

            doc_ids = [Encoder.tokenizer.encode(x) for x in data]

            if i == 0 and len(doc_ids) < 6:
                return None, None, 0

            if i != 0:
                if len(doc_ids) < (sum(stat_map[n][2][2]) / 4) + 1:
                    return None, None, 0

            if min([len(x) for x in doc_ids]) < 5:
                return None, None, 0
            
            all_doc_ids.append(doc_ids)

        doc_ids = all_doc_ids[0]
        neg_doc_ids = all_doc_ids[1:]

        # build pairs
        pairs = []
        labels = []

        number_map = [
            50, # 一
            230,
            156,
            349,
            443,
            803,
            950,
            1031, # 八
            1189,
            1320, # 十
            2907, 
            2117,
            3881,
            4489,
            2881,
            3871,
            4913
        ]

        for i, sent_ids in enumerate(doc_ids):
            # Truth
            p = 0
            if i == 0:
                p = i + 1
            elif i == len(doc_ids) - 1:
                p = i - 1
            else:
                if random.random() < 0.5:
                    p = i + 1
                else:
                    p = i - 1
            p_doc = doc_ids[p][:stat_map[n][1]]

            n_easy = []
            for _ in range(stat_map[n][2][1]):
                r = i
                while r == i or r == i - 1 or r == i + 1 or r in n_easy:
                       r = random.randint(0, len(doc_ids) - 1)
                n_easy.append(r)

            n_easy_doc = [doc_ids[x][:stat_map[n][1]] for x in n_easy]

            n_hard_doc = []
            for neg, num_try in zip(neg_doc_ids, stat_map[n][2][2]):
                tmp = []
                while len(tmp) < num_try:
                    r = random.randint(0, len(neg) - 1)
                    if r not in tmp:
                        n_hard_doc.append(neg[r][:stat_map[n][1]])
                        tmp.append(r)

            options = [p_doc] + n_easy_doc + n_hard_doc
            random.shuffle(options)

            label = options.index(p_doc)

            pair = sent_ids[:stat_map[n][0]] + [19]
            for i, option in enumerate(options):
                pair.extend([number_map[i], 20] + option + [18])
            pair += [19] + self.tokenizer.encode("正确的选项是：") + [Encoder.tokenizer.get_sentinel_id(0)]

            label_ids = [1, Encoder.tokenizer.get_sentinel_id(0)] + [number_map[label]] + [Encoder.tokenizer.get_sentinel_id(1)]

            pairs.append(pair)
            labels.append(label_ids)

        return pairs, labels, len(line)


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', default="../../../data/pretrain/pretrain_raw_10m.txt", type=str, help='Path to input TXT')
    group.add_argument('--input_neg1', default="../../../data/pretrain/pretrain_raw_10m_shuf.txt", type=str, help='Path to input TXT')
    group.add_argument('--input_neg2', default="../../../data/pretrain/pretrain_raw_10m_shuf_2.txt", type=str, help='Path to input TXT')
    group.add_argument('--input_neg3', default="../../../data/pretrain/pretrain_raw_10m_shuf_3.txt", type=str, help='Path to input TXT')
    group.add_argument('--input_neg4', default="../../../data/pretrain/pretrain_raw_10m_shuf_3.txt", type=str, help='Path to input TXT')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer_path', default="../../bpe_cn", type=str, help='Path of tokenizer')

    group = parser.add_argument_group(title='output data')
    group.add_argument("--output_path", default="../../pretrain_data", type=str)
    group.add_argument('--output_prefix', default="nss_small", type=str,
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset_impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes to launch')
    group.add_argument('--log_interval', type=int, default=10000,
                       help='Interval between progress updates')

    args = parser.parse_args()
    args.keep_empty = False

    args.rank = 0
    args.make_vocab_size_divisible_by = 128

    return args

def main():
    args = get_args()
    startup_start = time.time()

    print("Opening", args.input)
    fin = open(args.input, 'r', encoding='utf-8')
    fin_neg1 = open(args.input_neg1, "r", encoding='utf-8')
    fin_neg2 = open(args.input_neg2, "r", encoding='utf-8')
    fin_neg3 = open(args.input_neg3, "r", encoding='utf-8')
    fin_neg4 = open(args.input_neg4, "r", encoding='utf-8')

    encoder = Encoder(args)
    tokenizer = EncDecTokenizer(os.path.join(args.tokenizer_path, 'vocab.txt'))
    # pool = Pool(args.workers, initializer=encoder.initializer)
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
    
    # use the tokenizer to encode the sentences
    encoded_docs = pool.imap_unordered(encoder.encode, zip(fin, fin_neg1, fin_neg2, fin_neg3, fin_neg4), 10)

    level = "document"

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Output prefix: {args.output_prefix}")
    context_bin_file = os.path.join(args.output_path, "{}_{}_context.bin".format(args.output_prefix, level))
    context_idx_file = os.path.join(args.output_path,  "{}_{}_context.idx".format(args.output_prefix, level))
    target_bin_file = os.path.join(args.output_path,  "{}_{}_target.bin".format(args.output_prefix, level))
    target_idx_file = os.path.join(args.output_path,  "{}_{}_target.idx".format(args.output_prefix, level))
    
    builder_context = indexed_dataset.make_builder(context_bin_file, impl=args.dataset_impl, dtype=np.uint16)
    builder_target = indexed_dataset.make_builder(target_bin_file, impl=args.dataset_impl, dtype=np.uint16)

    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0
    print("Time to startup:", startup_end - startup_start)

    # sentinel_idx = tokenizer.vocab_size # start from the last token of the tokenizer
    print("tokenizer vocab size:", tokenizer.vocab_size)
    for i, (pair_ids, label_ids, bytes_processed) in enumerate(encoded_docs, start=1):
        if pair_ids is None or label_ids is None:
            continue
        total_bytes_processed += bytes_processed

        for pids, lids in zip(pair_ids, label_ids):
            builder_context.add_item(torch.IntTensor(pids))
            builder_target.add_item(torch.IntTensor(lids))
        
        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed / elapsed / 1024 / 1024
            print(f"Processed {i} documents",
                  f"({i/elapsed} docs/s, {mbs} MB/s).",
                  file=sys.stderr)

    builder_context.finalize(context_idx_file)
    builder_target.finalize(target_idx_file)

    pool.close()

if __name__ == '__main__':
    main()
