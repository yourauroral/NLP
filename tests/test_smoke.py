"""冒烟测试：验证预训练模型能加载并正确区分情感。

运行方式：
    pytest tests/
或：
    python tests/test_smoke.py

该测试会真正加载权重并跑前向，因此需要 PyTorch，以及 model/ 下的权重
（sentiment.safetensors 或原始 0.npy~14.npy，可用 download_weights.py 准备）。
"""
import os
import sys

import numpy as np
import pytest

# 确保可以从仓库根目录导入 encoder
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

WEIGHTS = os.path.join(ROOT, 'model', 'sentiment.safetensors')
SHARD0 = os.path.join(ROOT, 'model', '0.npy')   # 代表性的原始分片
HAS_NPY = os.path.exists(SHARD0)

# 没有任何形式的权重（safetensors 或 .npy）时整体跳过，而不是报错失败
pytestmark = pytest.mark.skipif(
    not (os.path.exists(WEIGHTS) or HAS_NPY),
    reason='预训练权重缺失（需 model/sentiment.safetensors 或 0.npy~14.npy），跳过冒烟测试',
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


@pytest.mark.skipif(not HAS_NPY, reason='缺少原始 .npy 分片，跳过 .npy 回退测试')
def test_npy_fallback_matches_safetensors():
    """safetensors 缺失时，Model 应自动用同目录的原始 .npy 就地组装（无需下载）。"""
    from encoder import Model, SENTIMENT_NEURON

    # 指向一个不存在的 safetensors，强制走 .npy 回退（同目录下有 0.npy~14.npy）
    missing = os.path.join(ROOT, 'model', '__no_such_weights__.safetensors')
    assert not os.path.exists(missing)

    texts = ['I love this, it is wonderful!', 'I hate this, it is terrible!']
    feats = Model(weights_path=missing).transform(texts)

    assert feats.shape == (2, 4096)
    assert np.isfinite(feats).all()
    # 情感神经元仍能区分正负
    assert feats[0, SENTIMENT_NEURON] > feats[1, SENTIMENT_NEURON]

    # 若 safetensors 也在场，两种来源应给出一致结果
    if os.path.exists(WEIGHTS):
        ref = Model(weights_path=WEIGHTS).transform(texts)
        assert np.allclose(feats, ref, atol=1e-5)


if __name__ == '__main__':
    sys.exit(pytest.main([__file__, '-v']))
