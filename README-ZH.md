<div align="center">

<h1>🛍 ModelCenter</h1>

**高效低资源的大模型实现**

</div>

<p align="center">
  <a href="#总览">总览</a> •
  <a href="#文档">文档</a> •
  <a href="#安装">安装</a> •
  <a href="#快速上手">快速上手</a> •
  <a href="#模型支持">模型支持</a> •
  <a href="./README.md" target="_blank">English</a>
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

## 最新动态

- 2022/03/16 [0.1.0](https://github.com/OpenBMB/ModelCenter/releases/tag/v0.0.1-beta) ModelCenter 公开发布了第一个稳定版本, 修复了一些模型性能上和显存占用上的问题.
- 2022/03/21 [0.0.1-beta](https://github.com/OpenBMB/ModelCenter/releases/tag/v0.0.1-beta) ModelCenter 公开发布了第一个 beta 版本.

## 总览

ModelCenter 基于 [OpenBMB/BMTrain](https://github.com/OpenBMB/BMTrain/) 实现了一系列经典的预训练语言模型。 ModelCenter 在模型实现上的宗旨是 高效、低资源与高可用性, 并且能够支持分布式的训练.

我们的主要优势有：

- 易用性：相比 Deepspeed, Megatron, 我们拥有更好更灵活的封装，且配置 python 环境容易, 训练代码与 pytorch 风格统一。
- 更高效的显存利用：模型占用显存较大时，可能会导致 GPU 的计算能力未被充分使用时显存占用就已经跑满。我们的实现可以将显存占用降低数倍，进而使用更大的 batch-size 对 GPU 的计算能力进行更充分的利用。
- 低资源的高效分布式训练：在 [OpenBMB/BMTrain](https://github.com/OpenBMB/BMTrain/) 的支持下，我们能够将 ZeRO3 的优化轻易地扩展至各大预训练语言模型，并在分布式训练的通信和调度上作出优化。

## 文档

我们的[文档](https://modelcenter.readthedocs.io/)提供了关于工具包的更多信息。

## 安装

### 1. 用 pip 安装 (推荐)

```shell
$ pip install model-center
```

### 2. 从源代码安装

```shell
$ git clone https://github.com/OpenBMB/ModelCenter.git
$ cd ModelCenter
$ pip install -r requirements.txt
$ python3 setup.py install
```

## 快速上手

在本节中，你将学习如何在一个分类数据集上微调[BERT](https://arxiv.org/abs/1810.04805)模型。

### 1. 初始化BMTrain后端
首先，你需要在代码开头引入`bmtrain`并使用`bmtrain.init_distributed()`。

```python
# init bmtrain backend
import bmtrain as bmt
bmt.init_distributed(seed=0)
```

### 2. 准备模型
接下来，你可以从`model_center`中获取预训练好的BERT模型，例如*bert-base-uncased*。由于我们是在一个分类任务上微调BERT模型，所以需要在最后一层后添加一个全连接层。

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

### 3. 准备数据集
下一步是准备数据集，用于训练和验证模型。这里，我们使用[SuperGLUE benchmark](https://super.gluebenchmark.com/)中的[BoolQ](https://github.com/google-research-datasets/boolean-questions)数据集。你需要下载该数据集，并将解压后的文件夹放在`your_path_to_dataset`路径下。

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

### 4. 训练模型
现在，在设置优化器、学习率调整策略和损失函数后，就可以开始训练模型了！这里，我们训练BERT模型5轮，并且在每轮训练结束后验证模型的性能。

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

        # 前向传播
        logits = model(input_ids, attention_mask)

        # 计算损失
        loss = loss_func(logits.view(-1, logits.shape[-1]), labels.view(-1))

        # 缩放损失以避免fp16精度下溢
        loss = optimizer.loss_scale(loss)

        # 反向传播
        loss.backward()

        # 梯度裁剪
        grad_norm = bmt.optim.clip_grad_norm(optimizer.param_groups, max_norm=10.0, scale = optimizer.scale, norm_type = 2)

        bmt.optim_step(optimizer, lr_scheduler)

        # 在分布式训练时，只在rank为0时打印信息
        # 使用bmt.sum_loss(loss)聚合所有进程上的损失
        bmt.print_rank(
            "loss: {:.4f} | lr: {:.4e}, scale: {:10.4f} | grad_norm: {:.4f} |".format(
                bmt.sum_loss(loss).item(),
                lr_scheduler.current_lr,
                int(optimizer.scale),
                grad_norm,
            )
        )

    # 验证模型的性能
    model.eval()
    with torch.no_grad():
        pd = [] # 预测结果
        gt = [] # 标签
        for data in dev_dataloader:
            input_ids = data["input_ids"]
            attention_mask = data["attention_mask"]
            labels = data["labels"]

            logits = model(input_ids, attention_mask)
            loss = loss_func(logits.view(-1, logits.shape[-1]), labels.view(-1))

            logits = logits.argmax(dim=-1)

            pd.extend(logits.cpu().tolist())
            gt.extend(labels.cpu().tolist())

        # 聚合所有进程上的结果
        pd = bmt.gather_result(torch.tensor(pd).int()).cpu().tolist()
        gt = bmt.gather_result(torch.tensor(gt).int()).cpu().tolist()

        # 计算指标
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(gt, pd)
        bmt.print_rank(f"accuracy: {acc*100:.2f}")
```

### 5. 运行代码
你可以使用PyTorch原生的分布式训练启动器来运行上述代码，根据你的PyTorch版本选择下列命令中的一个。

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

更多信息请参考PyTorch[官方文档](https://pytorch.org/docs/stable/distributed.html#launch-utility)。


## 模型支持

- [CPM: A Large-scale Generative Chinese Pre-trained Language Model.](https://arxiv.org/abs/2012.00413) Zhengyan Zhang, Xu Han, Hao Zhou, Pei Ke, Yuxian Gu, Deming Ye, Yujia Qin, Yusheng Su, Haozhe Ji, Jian Guan, Fanchao Qi, Xiaozhi Wang, Yanan Zheng, Guoyang Zeng, Huanqi Cao, Shengqi Chen, Daixuan Li, Zhenbo Sun, Zhiyuan Liu, Minlie Huang, Wentao Han, Jie Tang, Juanzi Li, Xiaoyan Zhu, Maosong Sun. 我们支持使用 ``CPM1.from_pretrained(identifier)`` 来加载下列模型：

    - cpm1-large

- [CPM-2: Large-scale Cost-efficient Pre-trained Language Models.](https://arxiv.org/abs/2106.10715) Zhengyan Zhang, Yuxian Gu, Xu Han, Shengqi Chen, Chaojun Xiao, Zhenbo Sun, Yuan Yao, Fanchao Qi, Jian Guan, Pei Ke, Yanzheng Cai, Guoyang Zeng, Zhixing Tan, Zhiyuan Liu, Minlie Huang, Wentao Han, Yang Liu, Xiaoyan Zhu, Maosong Sun. 我们支持使用 ``CPM2.from_pretrained(identifier)`` 来加载下列模型：

    - cpm2-large

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.](https://arxiv.org/abs/1810.04805) Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova. 我们支持使用 ``Bert.from_pretrained(identifier)`` 来加载下列模型：

    - bert-base-cased
    - bert-base-uncased
    - bert-large-cased
    - bert-large-uncased
    - bert-base-chinese
    - bert-base-multilingual-cased

- [T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) Colin Raffel and Noam Shazeer and Adam Roberts and Katherine Lee and Sharan Narang and Michael Matena and Yanqi Zhou and Wei Li and Peter J. Liu.. 我们支持使用 ``T5.from_pretrained(identifier)`` 来加载下列模型：

    - t5-small
    - t5-base
    - t5-large
    - t5-3b
    - t5-11b

- [GPT2: Language Models are Unsupervised Multitask Learners.](http://www.persagen.com/files/misc/radford2019language.pdf) Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 我们支持使用 ``GPT2.from_pretrained(identifier)`` 来加载下列模型：

    - gpt2-base
    - gpt2-medium
    - gpt2-large
    - gpt2-xl

- [GPT-J](https://github.com/kingoflolz/mesh-transformer-jax) (from EleutherAI) released in the repo [mesh-transformer-jax](https://github.com/kingoflolz/mesh-transformer-jax) by Ben Wang and Aran Komatsuzaki. 我们支持使用 ``GPTj.from_pretrained(identifier)`` 来加载下列模型：

    - gptj-6b

## 运行性能

你可以在 [OpenBMB/BMTrain](https://github.com/OpenBMB/BMTrain) 仓库中找到更多的性能测试效果.

## 开源社区

欢迎贡献者参照我们的[贡献指南](https://github.com/OpenBMB/ModelCenter/blob/main/CONTRIBUTING.md)贡献相关代码。

您也可以在其他平台与我们沟通交流:
- QQ群: 735930538
- 官方网站: http://www.openbmb.org
- 微博: http://weibo.cn/OpenBMB
- Twitter: https://twitter.com/OpenBMB

## 开源许可

该工具包使用[Apache 2.0](https://github.com/OpenBMB/ModelCenter/blob/main/LICENSE)开源许可证。