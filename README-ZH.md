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

## 模型支持

- [CPM: A Large-scale Generative Chinese Pre-trained Language Model.](https://arxiv.org/abs/2012.00413) Zhengyan Zhang, Xu Han, Hao Zhou, Pei Ke, Yuxian Gu, Deming Ye, Yujia Qin, Yusheng Su, Haozhe Ji, Jian Guan, Fanchao Qi, Xiaozhi Wang, Yanan Zheng, Guoyang Zeng, Huanqi Cao, Shengqi Chen, Daixuan Li, Zhenbo Sun, Zhiyuan Liu, Minlie Huang, Wentao Han, Jie Tang, Juanzi Li, Xiaoyan Zhu, Maosong Sun. 我们支持使用 ``CPM1.from_pretrained(identifier)`` 来加载下列模型：

    - cpm1-large

- [CPM-2: Large-scale Cost-efficient Pre-trained Language Models.](https://arxiv.org/abs/2106.10715) Zhengyan Zhang, Yuxian Gu, Xu Han, Shengqi Chen, Chaojun Xiao, Zhenbo Sun, Yuan Yao, Fanchao Qi, Jian Guan, Pei Ke, Yanzheng Cai, Guoyang Zeng, Zhixing Tan, Zhiyuan Liu, Minlie Huang, Wentao Han, Yang Liu, Xiaoyan Zhu, Maosong Sun. 我们支持使用 ``CPM2.from_pretrained(identifier)`` 来加载下列模型：

    - cpm2-large

- [Bert: Pre-training of Deep Bidirectional Transformers for Language Understanding.](https://arxiv.org/abs/1810.04805) Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova. 我们支持使用 ``Bert.from_pretrained(identifier)`` 来加载下列模型：

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

你可以在 [BMTrain](https://github.com/OpenBMB/BMTrain) 仓库中找到更多的性能测试效果.

## 开源社区

欢迎贡献者参照我们的[贡献指南](https://github.com/OpenBMB/ModelCenter/blob/main/CONTRIBUTING.md)贡献相关代码。

您也可以在其他平台与我们沟通交流:
- QQ群: 735930538
- 官方网站: http://www.openbmb.org
- 微博: http://weibo.cn/OpenBMB
- Twitter: https://twitter.com/OpenBMB

## 开源许可

该工具包使用[Apache 2.0](https://github.com/OpenBMB/ModelCenter/blob/main/LICENSE)开源许可证。