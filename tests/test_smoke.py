"""冒烟测试：验证预训练模型能加载并正确区分情感。

运行方式：
    pytest tests/
或：
    python tests/test_smoke.py

该测试会真正加载权重并跑前向，因此需要 PyTorch 与
model/sentiment.safetensors 就绪（可用 download_weights.py / convert_weights.py 准备）。
"""
import os
import sys

import numpy as np
import pytest

# 确保可以从仓库根目录导入 encoder
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

WEIGHTS = os.path.join(ROOT, 'model', 'sentiment.safetensors')

# 缺少权重文件时跳过，而不是报错失败
pytestmark = pytest.mark.skipif(
    not os.path.exists(WEIGHTS),
    reason='预训练权重缺失 (model/sentiment.safetensors)，跳过冒烟测试',
)


def test_transform_shape_and_sentiment():
    """transform 输出形状正确，且情感神经元能区分正负文本。"""
    from encoder import Model, SENTIMENT_NEURON

    model = Model(weights_path=WEIGHTS)
    feats = model.transform(['I love this, it is wonderful!',
                             'I hate this, it is terrible!'])

    # 形状应为 [样本数, 4096]
    assert feats.shape == (2, 4096)
    assert np.isfinite(feats).all()

    # 正面文本在情感神经元上的激活应高于负面文本
    pos, neg = feats[0, SENTIMENT_NEURON], feats[1, SENTIMENT_NEURON]
    assert pos > neg, f'情感神经元未能区分情感: pos={pos:.4f} neg={neg:.4f}'


def test_two_instances_are_independent():
    """两次实例化互不干扰，且结果确定可复现。"""
    from encoder import Model

    m1 = Model(weights_path=WEIGHTS)
    m2 = Model(weights_path=WEIGHTS)

    text = ['a neutral sentence about the weather today']
    f1 = m1.transform(text)
    f2 = m2.transform(text)

    # 相同输入、相同权重 → 两个实例应给出一致结果
    assert np.allclose(f1, f2, atol=1e-4)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
