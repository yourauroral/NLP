import os
import html
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

def train_with_reg_cv(trX, trY, vaX, vaY, teX=None, teY=None, penalty='l1',
        C=2**np.arange(-8, 1).astype(float), seed=42):
    """
    交叉验证训练逻辑回归分类器

    Args:
        trX, trY: 训练集特征和标签
        vaX, vaY: 验证集特征和标签
        teX, teY: 测试集特征和标签 (可选)
        penalty: 正则化类型 ('l1'或'l2')
        C: 正则化强度候选值数组
        seed: 随机种子

    Returns:
        score: 测试准确率 (百分比)
        c: 最优正则化系数
        nnotzero: 非零特征数量
    """
    scores = []
    # 在不同的C值上进行交叉验证
    for i, c in enumerate(C):
        model = LogisticRegression(C=c, penalty=penalty, solver='liblinear', random_state=seed+i)
        model.fit(trX, trY)
        score = model.score(vaX, vaY)  # 验证集上的准确率
        scores.append(score)

    # 选择最优的C值
    c = C[np.argmax(scores)]

    # 用最优参数重新训练模型
    model = LogisticRegression(C=c, penalty=penalty, solver='liblinear', random_state=seed+len(C))
    model.fit(trX, trY)

    # 计算非零系数数量 (L1正则化会产生稀疏解)
    nnotzero = np.sum(model.coef_ != 0)

    # 在测试集或验证集上评估最终性能
    if teX is not None and teY is not None:
        score = model.score(teX, teY)*100.
    else:
        score = model.score(vaX, vaY)*100.

    return score, c, nnotzero


def load_sst(path):
    """加载Stanford Sentiment Treebank CSV，返回 (句子列表, 标签数组)"""
    data = pd.read_csv(path)
    X = data['sentence'].values.tolist()
    Y = data['label'].values
    return X, Y


def sst_binary(data_dir='data/'):
    """
    加载Stanford Sentiment Treebank二分类数据集

    与标准预处理版本不同，这里使用原始文本而非tokenized版本
    返回 trX/vaX/teX 文本与 trY/vaY/teY 标签 (0=负面, 1=正面)
    """
    trX, trY = load_sst(os.path.join(data_dir, 'train_binary_sent.csv'))
    vaX, vaY = load_sst(os.path.join(data_dir, 'dev_binary_sent.csv'))
    teX, teY = load_sst(os.path.join(data_dir, 'test_binary_sent.csv'))
    return trX, vaX, teX, trY, vaY, teY


def preprocess(text, front_pad='\n ', end_pad=' '):
    """
    文本预处理函数 - 为模型输入准备文本

    Args:
        text: 输入文本字符串
        front_pad: 前缀填充字符
        end_pad: 后缀填充字符

    Returns:
        预处理后的字节序列
    """
    text = html.unescape(text)           # 解码HTML实体
    text = text.replace('\n', ' ').strip()  # 替换换行符并去除首尾空格
    text = front_pad + text + end_pad    # 添加前后缀标记
    text = text.encode()                 # 转换为字节序列
    return text


def iter_data(*data, **kwargs):
    """将一个或多个等长数组按 size (默认128) 分批生成"""
    size = kwargs.get('size', 128)

    try:
        n = len(data[0])
    except:
        n = data[0].shape[0]

    batches = n // size
    if n % size != 0:
        batches += 1

    for b in range(batches):
        start = b * size
        end = (b + 1) * size
        if end > n:
            end = n

        if len(data) == 1:
            yield data[0][start:end]
        else:
            yield tuple([d[start:end] for d in data])


class HParams(object):
    """
    超参数容器类 - 用于存储和管理模型超参数

    使用方式:
        hps = HParams(learning_rate=0.001, batch_size=32)
        print(hps.learning_rate)  # 0.001
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)