# -*- coding: utf-8 -*-
# 配置参数

class TrainingConfig(object):
    epoches = 10
    evaluateEvery = 20
    checkpointEvery = 100
    learningRate = 0.001
    
    
class ModelConfig(object):
    embeddingSize = 200
    
    filters = 128  # 内层一维卷积核的数量，外层卷积核的数量应该等于embeddingSize，因为要确保每个layer后的输出维度和输入维度是一致的。
    numHeads = 8  # Attention 的头数
    numBlocks = 5  # 设置transformer block的数量
    epsilon = 1e-8  # LayerNorm 层中的最小除数
    keepProp = 0.9  # multi head attention 中的dropout
    
    dropoutKeepProb = 0.5 # 全连接层的dropout
    l2RegLambda = 0.0
    
    
class Config(object):
    sequenceLength = 128  # 取了所有序列长度的均值
    batchSize = 128
    
    trainPath = 'data/train.txt'
    testPath = 'data/test.txt'
    dataSource = "../data/compare.txt"
    
    private_store_pos = "2-5"
    checkpointSavePath = "../model/transformer/"+ private_store_pos +"/model/my-model"
    savedModelPath = "../model/transformer/"+ private_store_pos +"/savedModel"
    summarySavePath = "summary/" + private_store_pos
    train_result_index = "../model/transformer/"+ private_store_pos +"/train_re.json"
    dev_result_index = "../model/transformer/"+ private_store_pos +"/dev_re.json"
    #wordPiecePos = "../vocab.pickle"
    
    stopWordSource = ""
    
    numClasses = 6  # 二分类设置为1，多分类设置为类别的数目
    
    rate = 0.8  # 训练集的比例
    
    training = TrainingConfig()
    
    model = ModelConfig()
    
if __name__ == '__main__':
    # 实例化配置参数对象
    config = Config()

    
