# 生成评论与发现情感 (Generating Reviews and Discovering Sentiment)

> **本项目在保留核心功能的前提下，已部分重写以适配新版本依赖（TensorFlow 2.x）。**

本仓库复现论文 [《Learning to Generate Reviews and Discovering Sentiment》](https://arxiv.org/abs/1704.01444)（Alec Radford、Rafal Jozefowicz、Ilya Sutskever）的代码。

论文的核心发现是：在海量亚马逊评论上以无监督方式训练的字符级语言模型，会自发学习到一个**“情感神经元”**——单个隐藏单元（本模型中为第 **2388** 维）即可高精度地表征文本情感，无需任何情感标注。

目前代码支持将该语言模型用作**特征提取器**。

## 环境要求

| 依赖 | 版本 |
| --- | --- |
| TensorFlow | 2.9.0 |
| Python | 3.8 (Ubuntu 20.04) |
| CUDA | 11.2 |
| GPU | RTX 3090 × 1 |

> 说明：模型基于 TensorFlow 1.x 的静态图编写，`encoder.py` 顶部通过 `tf.compat.v1` 兼容层（`disable_v2_behavior()`）在 TF 2.x 上运行原始计算图。

## 快速开始

将文本转换为 4096 维特征向量：

```python
from encoder import Model

model = Model()
text = ['demo!']
text_features = model.transform(text)
```

`transform` 返回形状为 `[样本数, 4096]` 的特征矩阵，即 mLSTM 的最终隐藏状态。

## 情感分类示例

`sst_binary_demo.py` 演示了如何利用提取的特征，在 **Stanford Sentiment Treebank (SST)** 二分类数据集上复现论文中的情感分类结果。该示例还会像论文 Figure 3 一样，可视化情感神经元在正/负样本上的激活分布。

运行：

```bash
python sst_binary_demo.py
```

流程为：用预训练 mLSTM 提取特征 → 训练带 L1 正则、交叉验证选参的逻辑回归分类器 → 输出测试准确率，并绘制情感神经元激活值的直方图。

## 目录结构

```
.
├── encoder.py           # mLSTM 模型与特征提取（Model 类）
├── utils.py             # 数据加载、预处理、逻辑回归训练等工具函数
├── sst_binary_demo.py   # SST 二分类示例 + 情感神经元可视化
├── model/               # 预训练权重（0.npy ~ 14.npy）
└── data/                # SST 二分类数据集（train/dev/test_binary_sent.csv）
```

## 预训练模型

本仓库包含一个 **4096 单元的乘性 LSTM（multiplicative LSTM, mLSTM）** 模型的预训练参数，训练数据为 McAuley 等人 (2015) [1] 提出的亚马逊商品评论数据集。该数据集去重后包含 1996 年 5 月至 2014 年 7 月间超过 **8200 万** 条商品评论，对应超过 **380 亿** 训练字节。模型在 **四块 NVIDIA Pascal GPU** 上训练了**一个月**，处理速度约为每秒 12,500 个字符。

## 参考文献

[1] McAuley, Julian, Pandey, Rahul, and Leskovec, Jure. *Inferring networks of substitutable and complementary products.* In *Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 785–794. ACM, 2015.

## 许可证

本项目基于 [MIT License](LICENSE) 开源（Copyright © 2017 OpenAI）。
