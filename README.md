# 生成评论与发现情感 (Generating Reviews and Discovering Sentiment)

> **本项目已从原始 TensorFlow 实现重写为 PyTorch（仅推理 / 特征提取），权重沿用上游预训练参数。**

本仓库复现论文 [《Learning to Generate Reviews and Discovering Sentiment》](https://arxiv.org/abs/1704.01444)（Alec Radford、Rafal Jozefowicz、Ilya Sutskever）的代码。

论文的核心发现是：在海量亚马逊评论上以无监督方式训练的字符级语言模型，会自发学习到一个**“情感神经元”**——单个隐藏单元（本模型中为第 **2388** 维）即可高精度地表征文本情感，无需任何情感标注。

目前代码支持将该语言模型用作**特征提取器**。

## 环境要求

| 依赖 | 版本 |
| --- | --- |
| PyTorch | ≥ 2.0 |
| Python | ≥ 3.8 |
| GPU | 可选（实验环境为 AutoDL：RTX 3090 / CUDA 11.2） |

> 说明：模型为纯推理实现，`Model` 会自动选择 GPU（不可用时回退到 CPU）。
> 改用 PyTorch 后不再受上游 TensorFlow 对 Python 版本的 3.10 上限限制，可在 3.11 / 3.12 上运行。

## 快速开始

**1. 安装依赖**

```bash
pip install -r requirements.txt
```

**2. 准备预训练权重**

权重为单个文件 `sentiment.safetensors`（约 330 MB），不随仓库分发。运行：

```bash
python download_weights.py
```

该脚本**优先使用本地权重，必要时才下载**，顺序为：

1. `model/sentiment.safetensors` 已存在 → 直接使用；
2. 同目录下有原始 `0.npy ~ 14.npy` → 就地转换，无需下载；
3. 以上都没有 → 才从配置的地址下载。此时需在 `download_weights.py` 填好 `WEIGHTS_URL`（`sentiment.safetensors` 的完整地址），或设置环境变量 `export SENTIMENT_WEIGHTS_URL=<完整地址>`。

> `Model()` 同样只用本地权重：找不到 `sentiment.safetensors` 时会自动回退到同目录下的 `.npy`，全程不强制联网。

**3. 提取特征**

将文本转换为 4096 维特征向量：

```python
from encoder import Model

model = Model()
text = ['demo!']
text_features = model.transform(text)
```

`transform` 返回形状为 `[样本数, 4096]` 的特征矩阵，即 mLSTM 的最终细胞状态（cell state）。模型若检测到可用 GPU 会自动启用。

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
├── encoder.py           # mLSTM 模型与特征提取（PyTorch，Model 类）
├── utils.py             # 数据加载、预处理、逻辑回归训练等工具函数
├── sst_binary_demo.py   # SST 二分类示例 + 情感神经元可视化
├── download_weights.py  # 下载预训练权重（单个 safetensors）
├── convert_weights.py   # 把上游 15 个 .npy 合并为 sentiment.safetensors（一次性）
├── requirements.txt     # Python 依赖
├── tests/               # 冒烟测试
├── model/               # 预训练权重 sentiment.safetensors（需下载，不纳入版本控制）
└── data/                # SST 二分类数据集（train/dev/test_binary_sent.csv）
```

## 测试

```bash
pytest tests/
```

冒烟测试会加载权重、提取特征，并验证情感神经元能区分正/负文本。若 `model/` 下权重缺失，测试会自动跳过。

## 预训练模型

本项目使用一个 **4096 单元的乘性 LSTM（multiplicative LSTM, mLSTM）** 模型的预训练参数（通过 `download_weights.py` 获取），训练数据为 McAuley 等人 (2015) [1] 提出的亚马逊商品评论数据集。该数据集去重后包含 1996 年 5 月至 2014 年 7 月间超过 **8200 万** 条商品评论，对应超过 **380 亿** 训练字节。模型在 **四块 NVIDIA Pascal GPU** 上训练了**一个月**，处理速度约为每秒 12,500 个字符。

## 参考文献

[1] McAuley, Julian, Pandey, Rahul, and Leskovec, Jure. *Inferring networks of substitutable and complementary products.* In *Proceedings of the 21th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 785–794. ACM, 2015.

## 许可证

本项目基于 [MIT License](LICENSE) 开源（Copyright © 2017 OpenAI）。
