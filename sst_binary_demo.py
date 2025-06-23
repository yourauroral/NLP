# Stanford Sentiment Treebank 情感分析实验
# 复现 OpenAI "Learning to Generate Reviews and Discovering Sentiment" 论文

from encoder import Model                              
from matplotlib import pyplot as plt                  
from utils import sst_binary, train_with_reg_cv       

# 初始化预训练模型 (在82M Amazon评论上训练的mLSTM)
model = Model()

# 加载Stanford Sentiment Treebank数据集
# trX, vaX, teX: 训练/验证/测试文本
# trY, vaY, teY: 对应的情感标签 (0=负面, 1=正面)
trX, vaX, teX, trY, vaY, teY = sst_binary()

# 使用预训练模型提取4096维特征向量
# 将文本转换为稠密的数值表示，包含丰富的语义信息
trXt = model.transform(trX)    # 训练集特征
vaXt = model.transform(vaX)    # 验证集特征  
teXt = model.transform(teX)    # 测试集特征

# 训练逻辑回归分类器进行情感分类
# 使用L1正则化和交叉验证选择最优超参数
# full_rep_acc: 测试集准确率, c: 最优正则化系数, nnotzero: 使用的特征数
full_rep_acc, c, nnotzero = train_with_reg_cv(trXt, trY, vaXt, vaY, teXt, teY)

# 输出分类结果
print('%05.2f test accuracy'%full_rep_acc)      # 测试准确率
print('%05.2f regularization coef'%c)          # 正则化系数
print('%05d features used'%nnotzero)           # 有效特征数量

# 可视化情感神经元 (第2388个特征)
# 该神经元在无监督训练中自动学会了表示情感
sentiment_unit = trXt[:, 2388]                  # 提取情感神经元激活值

# 绘制正面和负面样本在情感神经元上的分布
plt.hist(sentiment_unit[trY==0], bins=25, alpha=0.5, label='neg')  # 负面样本分布
plt.hist(sentiment_unit[trY==1], bins=25, alpha=0.5, label='pos')  # 正面样本分布
plt.legend()
plt.show()