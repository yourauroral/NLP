#!/usr/bin/env python3
"""将原始的 15 个 .npy 权重合并、打包成单个 safetensors 文件。

原始权重以 0.npy ~ 14.npy 的形式分散保存（来自上游 TensorFlow 实现）。
本脚本按 mLSTM 的参数布局把它们组装成 12 个命名张量，保存为
`model/sentiment.safetensors` —— 单文件、加载快、跨框架安全。

这是一次性转换：仅当你手头有原始 .npy 分片时运行一次，
之后只需分发/上传生成的 sentiment.safetensors。

用法：
    python convert_weights.py                        # model/*.npy -> model/sentiment.safetensors
    python convert_weights.py <src_dir> <out_file>   # 自定义输入目录 / 输出文件
"""
import os
import sys

import numpy as np
import torch
from safetensors.torch import save_file

NUM_SHARDS = 15

# 各命名张量的来源 .npy 与期望形状（仅用于校验）。
# 该映射依据上游 TF 计算图中 get_variable 的调用顺序推导得到。
EXPECTED_SHAPES = {
    'embedding': (256, 64),      # 字节嵌入
    'wx': (64, 16384),           # 输入 -> 4 个门
    'wh': (4096, 16384),         # 隐状态 -> 4 个门（由 2,3,4,5 拼接）
    'wmx': (64, 4096),           # 乘性连接：输入侧
    'wmh': (4096, 4096),         # 乘性连接：隐状态侧
    'b': (16384,),               # 门偏置
    'gx': (16384,), 'gh': (16384,),    # 权重归一化增益（对应 wx / wh）
    'gmx': (4096,), 'gmh': (4096,),    # 权重归一化增益（对应 wmx / wmh）
    'out_w': (4096, 256), 'out_b': (256,),  # 输出投影（特征提取用不到，保留以求完整）
}


def assemble(src_dir):
    """读取 0.npy ~ 14.npy，组装为 {名称: torch.Tensor} 字典并校验形状。"""
    missing = [i for i in range(NUM_SHARDS)
               if not os.path.exists(os.path.join(src_dir, '%d.npy' % i))]
    if missing:
        sys.exit('✗ 缺少原始权重分片：%s（在 %s/ 下未找到）'
                 % (', '.join('%d.npy' % i for i in missing), src_dir))

    p = [np.load(os.path.join(src_dir, '%d.npy' % i)) for i in range(NUM_SHARDS)]

    raw = {
        'embedding': p[0],
        'wx':  p[1],
        'wh':  np.concatenate(p[2:6], axis=1),   # 四块 4096x4096 沿列拼成 4096x16384
        'wmx': p[6],
        'wmh': p[7],
        'b':   p[8],
        'gx':  p[9],  'gh':  p[10],
        'gmx': p[11], 'gmh': p[12],
        'out_w': p[13], 'out_b': p[14],
    }

    tensors = {}
    for k, arr in raw.items():
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        if arr.shape != EXPECTED_SHAPES[k]:
            sys.exit('✗ %s 形状为 %s，预期 %s' % (k, arr.shape, EXPECTED_SHAPES[k]))
        tensors[k] = torch.from_numpy(arr)
    return tensors


def main():
    src_dir = sys.argv[1] if len(sys.argv) > 1 else 'model'
    out_file = (sys.argv[2] if len(sys.argv) > 2
                else os.path.join('model', 'sentiment.safetensors'))

    tensors = assemble(src_dir)

    os.makedirs(os.path.dirname(out_file) or '.', exist_ok=True)
    save_file(tensors, out_file, metadata={
        'model': 'sentiment-neuron mLSTM (4096-unit, byte-level)',
        'sentiment_neuron': '2388',
    })

    n_params = sum(t.numel() for t in tensors.values())
    size_mb = os.path.getsize(out_file) / 1024 / 1024
    print('✓ 已写出 %s' % out_file)
    print('  张量 %d 个 · 参数 %d · 体积 %.1f MB' % (len(tensors), n_params, size_mb))


if __name__ == '__main__':
    main()
