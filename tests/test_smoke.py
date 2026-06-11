"""冒烟测试：验证预训练模型能加载并正确区分情感。

运行方式：
    pytest tests/
或：
    python tests/test_smoke.py

该测试会真正加载 model/ 下的权重并构建计算图，因此需要
TensorFlow 与全部预训练 .npy 文件就绪。
"""
import os
import sys

import numpy as np
import pytest

# 确保可以从仓库根目录导入 encoder
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODEL_DIR = os.path.join(ROOT, 'model')

# 缺少权重文件时跳过，而不是报错失败
pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(MODEL_DIR, '0.npy')),
    reason='预训练权重缺失 (model/0.npy)，跳过冒烟测试',
)


def test_transform_shape_and_sentiment():
    """transform 输出形状正确，且情感神经元能区分正负文本。"""
    from encoder import Model, SENTIMENT_NEURON

    model = Model(model_dir=MODEL_DIR)
    feats = model.transform(['I love this, it is wonderful!',
                             'I hate this, it is terrible!'])

    # 形状应为 [样本数, 4096]
    assert feats.shape == (2, 4096)
    assert np.isfinite(feats).all()

    # 正面文本在情感神经元上的激活应高于负面文本
    pos, neg = feats[0, SENTIMENT_NEURON], feats[1, SENTIMENT_NEURON]
    assert pos > neg, f'情感神经元未能区分情感: pos={pos:.4f} neg={neg:.4f}'


def test_two_instances_are_independent():
    """两次实例化互不干扰（验证全局状态问题已修复）。"""
    from encoder import Model

    m1 = Model(model_dir=MODEL_DIR)
    m2 = Model(model_dir=MODEL_DIR)

    text = ['a neutral sentence about the weather today']
    f1 = m1.transform(text)
    f2 = m2.transform(text)

    # 相同输入、相同权重 → 两个实例应给出一致结果
    assert np.allclose(f1, f2, atol=1e-4)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
