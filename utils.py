import os
import html
import numpy as np
import pandas as pd
import tensorflow as tf
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
    """
    加载Stanford Sentiment Treebank数据文件
    
    Args:
        path: CSV文件路径
    
    Returns:
        X: 句子列表
        Y: 对应的标签数组
    """
    data = pd.read_csv(path)
    X = data['sentence'].values.tolist()  # 提取句子文本
    Y = data['label'].values              # 提取情感标签
    return X, Y


def sst_binary(data_dir='generating-reviews-discovering-sentiment/data/'):
    """
    加载Stanford Sentiment Treebank二分类数据集
    
    与标准预处理版本不同，这里使用原始文本而非tokenized版本
    这样可以测试模型对原始文本的理解能力
    
    Args:
        data_dir: 数据文件夹路径
    
    Returns:
        trX, vaX, teX: 训练/验证/测试集文本
        trY, vaY, teY: 训练/验证/测试集标签 (0=负面, 1=正面)
    """
    trX, trY = load_sst(os.path.join(data_dir, 'train_binary_sent.csv'))
    vaX, vaY = load_sst(os.path.join(data_dir, 'dev_binary_sent.csv'))
    teX, teY = load_sst(os.path.join(data_dir, 'test_binary_sent.csv'))
    return trX, vaX, teX, trY, vaY, teY


def find_trainable_variables(key):
    """
    查找包含特定关键字的可训练变量
    
    Args:
        key: 变量名中包含的关键字
    
    Returns:
        匹配的TensorFlow变量列表
    """
    return tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, ".*{}.*".format(key))


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
    """
    数据批次迭代器 - 将数据分批处理
    
    Args:
        *data: 要分批的数据 (可以是多个数组)
        size: 批次大小 (默认128)
    
    Yields:
        每个批次的数据
    """
    size = kwargs.get('size', 128)
    
    # 获取数据长度
    try:
        n = len(data[0])
    except:
        n = data[0].shape[0]
    
    # 计算批次数量
    batches = n // size
    if n % size != 0:
        batches += 1

    # 生成批次数据
    for b in range(batches):
        start = b * size
        end = (b + 1) * size
        if end > n:
            end = n
        
        # 返回单个或多个数组的批次
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
        """
        初始化超参数
        
        Args:
            **kwargs: 任意关键字参数，会被设置为类的属性
        """
        for k, v in kwargs.items():
            setattr(self, k, v)  # 将每个参数设置为类属性