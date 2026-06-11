"""情感神经元 mLSTM —— PyTorch 推理实现。

在海量亚马逊评论上无监督训练的字节级乘性 LSTM（4096 单元）。
其第 2388 维隐单元自发学到了“情感”，可作为强特征用于情感分类。

本文件仅做推理（特征提取），权重为预训练且固定，因此：
  * 权重归一化在加载时一次性折叠进有效权重；
  * 前向全程 no_grad。

公开接口（与原 TensorFlow 版保持一致）：
    model = Model()                      # 自动选择 cuda / cpu
    feats = model.transform(texts)       # -> np.ndarray [N, 4096]
"""
import os
import time

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file

from utils import preprocess

# 情感神经元所在维度（无监督训练中自发学到的情感单元）
SENTIMENT_NEURON = 2388

DEFAULT_WEIGHTS = os.path.join('model', 'sentiment.safetensors')


class mLSTM(nn.Module):
    """乘性 LSTM（multiplicative LSTM）—— 用乘性连接增强信息流的 LSTM 变体。

    权重以 buffer 形式保存（推理期固定，不参与训练）。加载时即把权重归一化
    （tf.nn.l2_normalize(w, axis=0) * g）折叠进 wx/wh/wmx/wmh，前向时直接使用。
    """

    def __init__(self, weights, nhidden=4096, nembd=64):
        super().__init__()
        self.nhidden = nhidden
        self.nembd = nembd

        # 字节嵌入 + 输出投影（out_* 仅为完整性保留，特征提取不使用）
        self.register_buffer('embedding', weights['embedding'])   # (256, 64)
        self.register_buffer('out_w', weights['out_w'])           # (4096, 256)
        self.register_buffer('out_b', weights['out_b'])           # (256,)

        # 折叠权重归一化：按列 L2 归一化后乘以增益 g
        self.register_buffer('wx',  self._wn(weights['wx'],  weights['gx']))   # (64, 16384)
        self.register_buffer('wh',  self._wn(weights['wh'],  weights['gh']))   # (4096, 16384)
        self.register_buffer('wmx', self._wn(weights['wmx'], weights['gmx']))  # (64, 4096)
        self.register_buffer('wmh', self._wn(weights['wmh'], weights['gmh']))  # (4096, 4096)
        self.register_buffer('b',   weights['b'])                              # (16384,)

    @staticmethod
    def _wn(w, g):
        """对应 tf.nn.l2_normalize(w, axis=0) * g。"""
        return w / w.norm(dim=0, keepdim=True) * g

    @torch.no_grad()
    def forward(self, X, mask):
        """逐时间步前向。

        Args:
            X:    (N, T) 字节 id（0~255）
            mask: (N, T, 1) 浮点掩码，1=真实 token，0=填充位
        Returns:
            cells: (N, T, nhidden) 每个时间步的细胞状态
            c:     (N, nhidden) 最终细胞状态（即特征向量）
            h:     (N, nhidden) 最终隐藏状态
        """
        N, T = X.shape
        words = self.embedding[X]                       # (N, T, nembd)
        c = X.new_zeros(N, self.nhidden, dtype=torch.float32)
        h = X.new_zeros(N, self.nhidden, dtype=torch.float32)

        cells = []
        for t in range(T):
            x = words[:, t, :]                          # (N, nembd)

            # 乘性连接：输入与隐状态的逐元素乘积
            m = (x @ self.wmx) * (h @ self.wmh)         # (N, nhidden)
            # 四个门的线性部分
            z = (x @ self.wx) + (m @ self.wh) + self.b  # (N, 4*nhidden)

            i, f, o, u = torch.split(z, self.nhidden, dim=1)
            i = torch.sigmoid(i)    # 输入门
            f = torch.sigmoid(f)    # 遗忘门
            o = torch.sigmoid(o)    # 输出门
            u = torch.tanh(u)       # 候选值

            ct = f * c + i * u
            ht = o * torch.tanh(ct)

            # 填充位保持旧状态（mask=0 时不更新），故最终状态对应最后一个真实 token
            mm = mask[:, t, :]                          # (N, 1)
            c = ct * mm + c * (1 - mm)
            h = ht * mm + h * (1 - mm)
            cells.append(c)

        cells = torch.stack(cells, dim=1)               # (N, T, nhidden)
        return cells, c, h


class Model(object):
    """封装预训练 mLSTM 的特征提取器。

    每个实例独立持有权重，可安全地多次实例化。
    """

    def __init__(self, weights_path=DEFAULT_WEIGHTS, device=None, batch_size=128):
        """
        Args:
            weights_path: sentiment.safetensors 路径；若该文件不存在，则自动回退到
                          同目录下的原始 0.npy~14.npy 就地组装（因此不强制下载）。
            device: 'cuda' / 'cpu'，默认自动选择（有 GPU 用 GPU）
            batch_size: 前向批大小
        """
        self.device = (torch.device(device) if device is not None
                       else torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.batch_size = batch_size

        weights = self._load_weights(weights_path)
        self.net = mLSTM(weights).to(self.device).eval()
        self.nhidden = self.net.nhidden

    @staticmethod
    def _load_weights(weights_path):
        """加载权重字典：优先读 safetensors；没有则用同目录下的原始 .npy 就地组装。

        这样只要本地有任一形式的权重即可使用，不强制下载或预先转换。
        """
        if os.path.exists(weights_path):
            return load_file(weights_path)              # {名称: CPU 张量}
        # 没有 safetensors 时，回退到本地原始 .npy（无需下载 / 预转换）
        from convert_weights import assemble, has_shards
        model_dir = os.path.dirname(weights_path) or '.'
        if has_shards(model_dir):
            print('• 未找到 %s，改用本地原始 .npy 就地组装…' % weights_path)
            return assemble(model_dir)
        raise FileNotFoundError(
            '未找到权重：%s（同目录下也没有原始 0.npy~14.npy）。\n'
            '请运行 `python download_weights.py`（会优先用本地权重，必要时才下载），'
            '或把权重放到该目录。' % weights_path)

    def _pad_batch(self, batch, T=None):
        """把一批字节串后向填充为 (n, T) 的张量，并给出掩码。"""
        if T is None:
            T = max(len(s) for s in batch)
        n = len(batch)
        X = torch.zeros(n, T, dtype=torch.long)
        mask = torch.zeros(n, T, 1, dtype=torch.float32)
        for i, s in enumerate(batch):
            X[i, :len(s)] = torch.tensor(list(s), dtype=torch.long)
            mask[i, :len(s), 0] = 1.0
        return X, mask

    def transform(self, xs, verbose=False):
        """把文本列表编码为 [N, 4096] 特征矩阵（最终细胞状态）。"""
        tstart = time.time()
        seqs = [preprocess(x) for x in xs]
        lens = np.asarray([len(s) for s in seqs])
        order = np.argsort(lens)                        # 按长度排序以减少填充浪费

        feats = np.zeros((len(seqs), self.nhidden), dtype=np.float32)
        for start in range(0, len(seqs), self.batch_size):
            idx = order[start:start + self.batch_size]  # 原始下标
            X, mask = self._pad_batch([seqs[i] for i in idx])
            _, c, _ = self.net(X.to(self.device), mask.to(self.device))
            feats[idx] = c.cpu().numpy()                # 按原始顺序散射回去

        if verbose:
            print('%0.3f seconds to transform %d examples'
                  % (time.time() - tstart, len(seqs)))
        return feats

    def cell_transform(self, xs, indexes=None):
        """返回每个时间步的细胞状态 (N, T, K)，用于细粒度分析（如逐字符可视化）。

        indexes 不为 None 时只取指定的若干维。
        """
        seqs = [preprocess(x) for x in xs]
        T = max(len(s) for s in seqs)                   # 统一到全局最大长度，便于拼接
        Fs = []
        for start in range(0, len(seqs), self.batch_size):
            batch = seqs[start:start + self.batch_size]
            X, mask = self._pad_batch(batch, T=T)
            cells, _, _ = self.net(X.to(self.device), mask.to(self.device))
            cells = cells.cpu().numpy()
            if indexes is not None:
                cells = cells[:, :, indexes]
            Fs.append(cells)
        return np.concatenate(Fs, axis=0)


if __name__ == '__main__':
    model = Model()

    texts = [
        'This movie is amazing and wonderful!',   # 正面
        'This movie is terrible and boring!',     # 负面
    ]
    feats = model.transform(texts, verbose=True)
    print('特征矩阵形状：%s ｜ 设备：%s' % (feats.shape, model.device))

    for i, (text, score) in enumerate(zip(texts, feats[:, SENTIMENT_NEURON])):
        label = '正面' if score > 0 else '负面'
        print('文本%d: %s (情感神经元激活值: %+.4f)  内容: %r' % (i + 1, label, score, text))
