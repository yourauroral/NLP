import time
import numpy as np
import tensorflow as tf

# TensorFlow 1.x 兼容性设置
tf.placeholder = tf.compat.v1.placeholder                        
tf.compat.v1.disable_v2_behavior()                               
tf.placeholder = tf.compat.v1.placeholder                        
tf.Session = tf.compat.v1.Session                                
tf.get_variable = tf.compat.v1.get_variable                      
tf.variable_scope = tf.compat.v1.variable_scope                  
tf.global_variables_initializer = tf.compat.v1.global_variables_initializer  
tf.train.AdamOptimizer = tf.compat.v1.train.AdamOptimizer        

from tqdm import tqdm
import joblib
from utils import HParams, preprocess, iter_data

# 全局变量 - 追踪参数加载
global nloaded
nloaded = 0


def load_params(shape, dtype, *args, **kwargs):
    """从预训练numpy数组中加载权重"""
    global nloaded
    nloaded += 1
    return params[nloaded - 1]


def embd(X, ndim, scope='embedding'):
    """词嵌入层 - 将词索引转换为稠密向量"""
    with tf.variable_scope(scope):
        embd = tf.get_variable("w", [hps.nvocab, ndim], initializer=load_params)
        h = tf.nn.embedding_lookup(embd, X)
        return h


def fc(x, nout, act, wn=False, bias=True, scope='fc'):
    """全连接层 - 线性变换 + 激活函数"""
    with tf.variable_scope(scope):
        nin = x.get_shape()[-1].value
        w = tf.get_variable("w", [nin, nout], initializer=load_params)
        
        if wn:  # 权重归一化
            g = tf.get_variable("g", [nout], initializer=load_params)
            w = tf.nn.l2_normalize(w, axis=0) * g
        
        z = tf.matmul(x, w)
        
        if bias:
            b = tf.get_variable("b", [nout], initializer=load_params)
            z = z + b
        
        h = act(z)
        return h


def mlstm(inputs, c, h, M, ndim, scope='lstm', wn=False):
    """乘性LSTM - 使用乘性连接增强信息流的LSTM变体"""
    nin = inputs[0].get_shape()[1].value
    
    with tf.variable_scope(scope):
        # 标准LSTM权重
        wx = tf.get_variable("wx", [nin, ndim * 4], initializer=load_params)
        wh = tf.get_variable("wh", [ndim, ndim * 4], initializer=load_params)
        
        # 乘性LSTM的关键创新 - 乘性权重
        wmx = tf.get_variable("wmx", [nin, ndim], initializer=load_params)
        wmh = tf.get_variable("wmh", [ndim, ndim], initializer=load_params)
        
        b = tf.get_variable("b", [ndim * 4], initializer=load_params)
        
        # 权重归一化参数
        if wn:
            gx = tf.get_variable("gx", [ndim * 4], initializer=load_params)
            gh = tf.get_variable("gh", [ndim * 4], initializer=load_params)
            gmx = tf.get_variable("gmx", [ndim], initializer=load_params)
            gmh = tf.get_variable("gmh", [ndim], initializer=load_params)

    # 应用权重归一化
    if wn:
        wx = tf.nn.l2_normalize(wx, axis=0) * gx
        wh = tf.nn.l2_normalize(wh, axis=0) * gh
        wmx = tf.nn.l2_normalize(wmx, axis=0) * gmx
        wmh = tf.nn.l2_normalize(wmh, axis=0) * gmh

    cs = []
    
    # 按时间步处理序列
    for idx, x in enumerate(inputs):
        # 乘性LSTM核心：输入和隐藏状态的元素级乘法
        m = tf.matmul(x, wmx) * tf.matmul(h, wmh)
        
        # 门控计算
        z = tf.matmul(x, wx) + tf.matmul(m, wh) + b
        
        # 分离四个门
        i, f, o, u = tf.split(z, 4, 1)
        i = tf.nn.sigmoid(i)  # 输入门
        f = tf.nn.sigmoid(f)  # 遗忘门
        o = tf.nn.sigmoid(o)  # 输出门
        u = tf.tanh(u)        # 候选值
        
        if M is not None:  # 处理变长序列的掩码
            ct = f * c + i * u
            ht = o * tf.tanh(ct)
            m = M[:, idx, :]
            c = ct * m + c * (1 - m)
            h = ht * m + h * (1 - m)
        else:
            c = f * c + i * u
            h = o * tf.tanh(c)
        
        inputs[idx] = h
        cs.append(c)
    
    cs = tf.stack(cs)
    return inputs, cs, c, h


def model(X, S, M=None, reuse=False):
    """主模型：词嵌入 + mLSTM + 输出层"""
    nsteps = X.get_shape()[1]
    cstart, hstart = tf.unstack(S, num=hps.nstates)
    
    with tf.variable_scope('model', reuse=reuse):
        # 词嵌入
        words = embd(X, hps.nembd)
        inputs = tf.unstack(words, nsteps, 1)
        
        # mLSTM处理 - 产生4096维特征表示
        hs, cells, cfinal, hfinal = mlstm(
            inputs, cstart, hstart, M, hps.nhidden, scope='rnn', wn=hps.rnn_wn)
        
        # 输出层
        hs = tf.reshape(tf.concat(hs, 1), [-1, hps.nhidden])
        logits = fc(hs, hps.nvocab, act=lambda x: x, wn=hps.out_wn, scope='out')
    
    states = tf.stack([cfinal, hfinal], 0)
    return cells, states, logits


def ceil_round_step(n, step):
    """向上取整到step的倍数"""
    return int(np.ceil(n/step)*step)


def batch_pad(xs, nbatch, nsteps):
    """批次填充 - 将变长序列填充为固定长度"""
    xmb = np.zeros((nbatch, nsteps), dtype=np.int32)
    mmb = np.ones((nbatch, nsteps, 1), dtype=np.float32)
    
    for i, x in enumerate(xs):
        l = len(x)
        npad = nsteps - l
        xmb[i, -l:] = list(x)
        mmb[i, :npad] = 0  # 填充位置的掩码设为0
    
    return xmb, mmb


class Model(object):
    """情感神经元模型 - 封装预训练的mLSTM模型"""

    def __init__(self, nbatch=32, nsteps=32):
        """初始化模型"""
        global hps
        hps = HParams(
            load_path='model_params/params.jl',
            nhidden=4096,      # LSTM隐藏层维度 - 产生4096维特征
            nembd=64,          # 词嵌入维度
            nsteps=nsteps,     # 序列长度
            nbatch=nbatch,     # 批次大小
            nstates=2,         # LSTM状态数
            nvocab=256,        # 词汇表大小
            out_wn=False,
            rnn_wn=True,
            rnn_type='mlstm',
            embd_wn=True,
        )
        
        # 加载预训练参数
        global params
        params = [np.load('generating-reviews-discovering-sentiment/model/%d.npy'%i) for i in range(15)]
        params[2] = np.concatenate(params[2:6], axis=1)
        params[3:6] = []

        # 构建计算图
        X = tf.placeholder(tf.int32, [None, hps.nsteps])
        M = tf.placeholder(tf.float32, [None, hps.nsteps, 1])
        S = tf.placeholder(tf.float32, [hps.nstates, None, hps.nhidden])
        
        cells, states, logits = model(X, S, M, reuse=False)

        # 创建会话
        sess = tf.Session()
        tf.global_variables_initializer().run(session=sess)

        def seq_rep(xmb, mmb, smb):
            """获取序列的最终状态表示"""
            return sess.run(states, {X: xmb, M: mmb, S: smb})

        def seq_cells(xmb, mmb, smb):
            """获取序列的所有时间步细胞状态"""
            return sess.run(cells, {X: xmb, M: mmb, S: smb})

        def transform(xs):
            """主要特征提取函数 - 将文本转换为4096维特征向量"""
            tstart = time.time()
            
            # 文本预处理
            xs = [preprocess(x) for x in xs]
            
            # 按长度排序以提高批次处理效率
            lens = np.asarray([len(x) for x in xs])
            sorted_idxs = np.argsort(lens)
            unsort_idxs = np.argsort(sorted_idxs)
            sorted_xs = [xs[i] for i in sorted_idxs]
            maxlen = np.max(lens)
            
            # 批次处理
            offset = 0
            n = len(xs)
            smb = np.zeros((2, n, hps.nhidden), dtype=np.float32)
            
            for step in range(0, ceil_round_step(maxlen, nsteps), nsteps):
                start = step
                end = step + nsteps
                
                xsubseq = [x[start:end] for x in sorted_xs]
                
                # 移除已处理完的空序列
                ndone = sum([x == b'' for x in xsubseq])
                offset += ndone
                xsubseq = xsubseq[ndone:]
                sorted_xs = sorted_xs[ndone:]
                nsubseq = len(xsubseq)
                
                if nsubseq == 0:
                    continue
                
                xmb, mmb = batch_pad(xsubseq, nsubseq, nsteps)
                
                for batch in range(0, nsubseq, nbatch):
                    start = batch
                    end = batch + nbatch
                    
                    batch_smb = seq_rep(
                        xmb[start:end], 
                        mmb[start:end],
                        smb[:, offset+start:offset+end, :])
                    
                    smb[:, offset+start:offset+end, :] = batch_smb
            
            # 提取最终特征（LSTM隐藏状态）
            features = smb[0, unsort_idxs, :]
            
            print('%0.3f seconds to transform %d examples' %
                  (time.time() - tstart, n))
            
            return features  # 返回[n, 4096]的特征矩阵

        def cell_transform(xs, indexes=None):
            """提取所有时间步的细胞状态 - 用于详细分析"""
            Fs = []
            xs = [preprocess(x) for x in xs]
            
            for xmb in tqdm(
                    iter_data(xs, size=hps.nbatch), ncols=80, leave=False,
                    total=len(xs)//hps.nbatch):
                
                smb = np.zeros((2, hps.nbatch, hps.nhidden))
                n = len(xmb)
                
                xmb, mmb = batch_pad(xmb, hps.nbatch, hps.nsteps)
                smb = sess.run(cells, {X: xmb, S: smb, M: mmb})
                smb = smb[:, :n, :]
                
                if indexes is not None:
                    smb = smb[:, :, indexes]
                
                Fs.append(smb)
            
            Fs = np.concatenate(Fs, axis=1).transpose(1, 0, 2)
            return Fs

        self.transform = transform
        self.cell_transform = cell_transform


if __name__ == '__main__':
    mdl = Model()
    
    texts = [
        'This movie is amazing and wonderful!',  # 正面文本
        'This movie is terrible and boring!'     # 负面文本
    ]
    
    # 提取4096维特征
    text_features = mdl.transform(texts)
    
    # 分析情感神经元（第2388维特征）
    sentiment_neuron = text_features[:, 2388]
    
    # 输出结果
    for i, (text, score) in enumerate(zip(texts, sentiment_neuron)):
        emotion = "正面" if score > 0 else "负面"
        print(f"文本{i+1}: {emotion} (激活值: {score:.4f})")
        print(f"  内容: '{text}'")