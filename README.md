<div align="center">

<h1><img src="docs/source/_static/images/logo.png" height="32px"/> ModelCenter</h1>

**Efficient Low-Resource Big Models Implementation**

</div>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#documentation">Documentation</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#supported-models">Supported Models</a> •
  <a href="./README-ZH.md" target="_blank">简体中文</a>
</p>

<p align="center">

<a href='https://modelcenter.readthedocs.io/en/latest/?badge=latest'>
    <img src='https://readthedocs.org/projects/modelcenter/badge/?version=latest' alt='Documentation Status' />
</a>

<a href="https://github.com/OpenBMB/ModelCenter/releases">
    <img alt="GitHub release (latest by date including pre-releases)" src="https://img.shields.io/github/v/release/OpenBMB/ModelCenter?include_prereleases">
</a>

<a href="https://github.com/OpenBMB/ModelCenter/blob/main/LICENSE">
    <img alt="GitHub" src="https://img.shields.io/github/license/OpenBMB/ModelCenter">
</a>

</p>

## What's New

- 2022/04/27 [**ModelCenter 0.1.1**](https://github.com/OpenBMB/ModelCenter/releases/tag/v0.1.1) support RoBERTa. 
- 2022/04/06 [**ModelCenter 0.1.0**](https://github.com/OpenBMB/ModelCenter/releases/tag/v0.1.0) ModelCenter has publicly released the first stable version, which fixes some bugs in model performance and GPU memory usage.
- 2022/03/21 [**ModelCenter 0.0.1-beta**](https://github.com/OpenBMB/ModelCenter/releases/tag/v0.0.1-beta) ModelCenter has publicly released the first beta version.

## Overview

ModelCenter implements pre-trained language models (PLMs) based on [OpenBMB/BMTrain](https://github.com/OpenBMB/BMTrain/) backend. ModelCenter supports Efficient, Low-Resource, Extendable model usage and distributed training.

Our main advantages are:

- **Easy to use**. Compared to Deepspeed and Megatron, we have better and more flexible code-packaging and easy to configure python environments, and the training code is uniform with PyTorch style.
- **More efficient memory utilization**. Models with large memory footprints can cause OOM (out of memory) before the computational power of the GPU is fully utilized. Our implementation reduces the memory footprint by several times, allowing more efficient use of the GPU's computational power with a larger batch size.
- **Efficient distributed training with low resources**. With the support of [OpenBMB/BMTrain](https://github.com/OpenBMB/BMTrain/), we are able to easily extend ZeRO3's optimization to any PLMs, and we optimize communication and time scheduling for faster distributed training.

## Documentation

Our [documentation](https://modelcenter.readthedocs.io/) provides more information about the package.

## Installation

### 1. From PyPI (Recommend)

```shell
$ pip install model-center
```

### 2. From Source

```shell
$ git clone https://github.com/OpenBMB/ModelCenter.git
$ cd ModelCenter
$ pip install -r requirements.txt
$ python3 setup.py install
```

## Quick Start

In the quick start, you will walk through how to fine-tune a [BERT](https://arxiv.org/abs/1810.04805) model on a classification task.

### 1. Initialize bmtrain backend
First, you need to import `bmtrain` and use `bmtrain.init_distributed()` at the beginning of your code, which can initialize the distributed environments. 

```python
import bmtrain as bmt
bmt.init_distributed(seed=0)
```

### 2. Prepare the model
Next, you can simply get a pre-trained BERT model from `model_center`, e.g., *bert-base-uncased*. When fine-tuning BERT on the classification task, a feed-forward layer need to be appended to the last layer.

```python
import torch
from model_center.model import Bert, BertConfig
from model_center.layer import Linear

class BertModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = Bert.from_pretrained("bert-base-uncased")
        self.dense = Linear(config.dim_model, 2)
        bmt.init_parameters(self.dense)

    def forward(self, input_ids, attention_mask):
        pooler_output = self.bert(input_ids=input_ids, attention_mask=attention_mask).pooler_output
        logits = self.dense(pooler_output)
        return logits

config = BertConfig.from_pretrained("bert-base-uncased")
model = BertModel(config)
```

If only config is needed instead of pretrained checkpoint, you can initialize a model as the following:

```
config = BertConfig.from_json_file("your/path/to/config.json")
model = Bert(config)
bmt.init_parameters(model)
# bmt.load(model, "your/path/to/pytorch_model.pt")
```

### 3. Perpare the dataset
The next step is to prepare the dataset used for training and evaluation. Here, we use the [BoolQ](https://github.com/google-research-datasets/boolean-questions) dataset from the [SuperGLUE benchmark](https://super.gluebenchmark.com/). You need to download the dataset and put the unzipped folder in `your_path_to_dataset`.

```python
from model_center.dataset.bertdataset import DATASET
from model_center.dataset import DistributedDataLoader
from model_center.tokenizer import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
splits = ['train', 'dev']
dataset = {}

for split in splits:
    dataset[split] = DATASET['BoolQ']('your_path_to_dataset', split, bmt.rank(), bmt.world_size(), tokenizer, max_encoder_length=512)

batch_size = 64
train_dataloader = DistributedDataLoader(dataset['train'], batch_size=batch_size, shuffle=True)
dev_dataloader = DistributedDataLoader(dataset['dev'], batch_size=batch_size, shuffle=False)
```

### 4. Train the model
Now, select optimizer, learning rate scheduler, loss function, and then start training the model! Here, we train BERT for 5 epochs and evaluate it at the end of each epoch.

```python
optimizer = bmt.optim.AdamOffloadOptimizer(model.parameters())

lr_scheduler = bmt.lr_scheduler.Noam(
    optimizer, 
    start_lr = 1e-5,
    warmup_iter = 100, 
    end_iter = -1)

loss_func = bmt.loss.FusedCrossEntropy(ignore_index=-100)

for epoch in range(5):
    model.train()
    for data in train_dataloader:
        input_ids = data['input_ids']
        attention_mask = data['attention_mask']
        labels = data['labels']

        optimizer.zero_grad()

        # model forward
        logits = model(input_ids, attention_mask)

        # calculate loss
        loss = loss_func(logits.view(-1, logits.shape[-1]), labels.view(-1))

        # use bmt.sum_loss(loss) to gather all loss information from all distributed processes
        global_loss = bmt.sum_loss(loss).item()

        # scale loss to avoid precision underflow of fp16
        loss = optimizer.loss_scale(loss)

        # model backward
        loss.backward()

        # clip gradient norm
        grad_norm = bmt.optim.clip_grad_norm(optimizer.param_groups, max_norm=10.0, scale = optimizer.scale, norm_type = 2)

        bmt.optim_step(optimizer, lr_scheduler)

        # print information only on rank 0 when distributed training
        bmt.print_rank(
            "loss: {:.4f} | lr: {:.4e}, scale: {:10.4f} | grad_norm: {:.4f} |".format(
                global_loss,
                lr_scheduler.current_lr,
                int(optimizer.scale),
                grad_norm,
            )
        )

    # evaluate model
    model.eval()
    with torch.no_grad():
        pd = [] # prediction
        gt = [] # ground_truth
        for data in dev_dataloader:
            input_ids = data["input_ids"]
            attention_mask = data["attention_mask"]
            labels = data["labels"]

            logits = model(input_ids, attention_mask)
            loss = loss_func(logits.view(-1, logits.shape[-1]), labels.view(-1))

            logits = logits.argmax(dim=-1)

            pd.extend(logits.cpu().tolist())
            gt.extend(labels.cpu().tolist())

        # gather results from all distributed processes
        pd = bmt.gather_result(torch.tensor(pd).int()).cpu().tolist()
        gt = bmt.gather_result(torch.tensor(gt).int()).cpu().tolist()

        # calculate metric
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(gt, pd)
        bmt.print_rank(f"accuracy: {acc*100:.2f}")
```

### 5. Run your code
You can run the above code using the same launch command as the distributed module of PyTorch.

Choose one of the following commands depending on your version of PyTorch.

* `${MASTER_ADDR}` means the IP address of the master node.
* `${MASTER_PORT}` means the port of the master node.
* `${NNODES}` means the total number of nodes.
* `${GPU_PER_NODE}` means the number of GPUs per node.
* `${NODE_RANK}` means the rank of this node.

#### torch.distributed.launch
```shell
$ python3 -m torch.distributed.launch --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node ${GPU_PER_NODE} --nnodes ${NNODES} --node_rank ${NODE_RANK} train.py
```

#### torchrun

```shell
$ torchrun --nnodes=${NNODES} --nproc_per_node=${GPU_PER_NODE} --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} train.py
```


For more information, please refer to the [documentation](https://pytorch.org/docs/stable/distributed.html#launch-utility).


## Supported Models


- CPM-1[^1]. We currently support loading the following checkpoint via ``CPM1.from_pretrained(identifier)`` of the following:

    - cpm1-large

- CPM-2[^2]. We currently support loading the following checkpoint via ``CPM2.from_pretrained(identifier)`` of the following:

    - cpm2-large

- BERT[^3]. We currently support loading the following checkpoint via ``Bert.from_pretrained(identifier)`` of the following:

    - bert-base-cased
    - bert-base-uncased
    - bert-large-cased
    - bert-large-uncased
    - bert-base-chinese
    - bert-base-multilingual-cased

- RoBERTa[^4]. We currently support loading the following checkpoint via ``Roberta.from_pretrained(identifier)`` of the following:

    - roberta-base
    - roberta-large

- T5[^5]. We currently support loading the following checkpoint via ``T5.from_pretrained(identifier)`` of the following:

    - t5-small
    - t5-base
    - t5-large
    - t5-3b
    - t5-11b

- GPT-2[^6]. We currently support loading the following checkpoint via ``GPT2.from_pretrained(identifier)`` of the following:

    - gpt2-base
    - gpt2-medium
    - gpt2-large
    - gpt2-xl

- GPT-J[^7]. We currently support loading the following checkpoint via ``GPTj.from_pretrained(identifier)`` of the following:

    - gptj-6b

## Performance

You can find more performance metrics in the repo [OpenBMB/BMTrain](https://github.com/OpenBMB/BMTrain).

## Community

We welcome everyone to contribute codes following our [contributing guidelines](https://github.com/OpenBMB/ModelCenter/blob/main/CONTRIBUTING.md).

You can also find us on other platforms:
- QQ Group: 735930538
- Website: https://www.openbmb.org
- Weibo: http://weibo.cn/OpenBMB
- Twitter: https://twitter.com/OpenBMB

## License

The package is released under the [Apache 2.0](https://github.com/OpenBMB/ModelCenter/blob/main/LICENSE) License.

## References

[^1]: [CPM: A Large-scale Generative Chinese Pre-trained Language Model.](https://arxiv.org/abs/2012.00413) Zhengyan Zhang, Xu Han, Hao Zhou, Pei Ke, Yuxian Gu, Deming Ye, Yujia Qin, Yusheng Su, Haozhe Ji, Jian Guan, Fanchao Qi, Xiaozhi Wang, Yanan Zheng, Guoyang Zeng, Huanqi Cao, Shengqi Chen, Daixuan Li, Zhenbo Sun, Zhiyuan Liu, Minlie Huang, Wentao Han, Jie Tang, Juanzi Li, Xiaoyan Zhu, Maosong Sun.

[^2]: [CPM-2: Large-scale Cost-efficient Pre-trained Language Models.](https://arxiv.org/abs/2106.10715) Zhengyan Zhang, Yuxian Gu, Xu Han, Shengqi Chen, Chaojun Xiao, Zhenbo Sun, Yuan Yao, Fanchao Qi, Jian Guan, Pei Ke, Yanzheng Cai, Guoyang Zeng, Zhixing Tan, Zhiyuan Liu, Minlie Huang, Wentao Han, Yang Liu, Xiaoyan Zhu, Maosong Sun.

[^3]: [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.](https://arxiv.org/abs/1810.04805) Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.

[^4]: [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, Veselin Stoyanov.

[^5]: [T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) Colin Raffel and Noam Shazeer and Adam Roberts and Katherine Lee and Sharan Narang and Michael Matena and Yanqi Zhou and Wei Li and Peter J. Liu.

[^6]: [GPT2: Language Models are Unsupervised Multitask Learners.](http://www.persagen.com/files/misc/radford2019language.pdf) Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever.

[^7]: [GPT-J](https://github.com/kingoflolz/mesh-transformer-jax) (from EleutherAI) released in the repo [mesh-transformer-jax](https://github.com/kingoflolz/mesh-transformer-jax) by Ben Wang and Aran Komatsuzaki.