# -*- coding: utf-8 -*-
import numpy as np      
import gensim
from Paras import Config
from tensorflow.contrib import learn
#import tensorflow as tf

class OtherDataSet():
    def __init__(self, config):
        self.config = config
        self._dataSource = config.dataSource
        self._stopWordSource = config.stopWordSource  
        
        self._sequenceLength = config.sequenceLength  # 每条输入的序列处理为定长
        self._embeddingSize = config.model.embeddingSize
        self._batchSize = config.batchSize
        self._rate = config.rate
        
        self._stopWordDict = {}
        
        self.trainReviews = []
        self.trainLabels = []
        
        self.evalReviews = []
        self.evalLabels = []
        
        self.wordEmbedding =None
        
        self.labelList = []
    
    def my_readData(self,filepath):
        '''从txt文件中读取每一条数据，分词提取
        return:
            sentences: [[word,word,...], [, , ...], ...]，
            label: [int,int,...]
        '''
        #print("beginning  reading...")
        sentences, labels = [], []
        with open(filepath,'r', encoding ='utf-8') as f:
            for line in f:
                new_sen = (line.split("||")[0] + line.split("||")[1]).split(" ")
                if len(new_sen)  >= self._sequenceLength: continue  
                sentences.append(new_sen + ["<PAD>"]*(self._sequenceLength - len(new_sen)))
                labels.append(int(line.split("||")[2].strip()))
        #print("ending  reading...")
        #for sen,lab in zip(sentences, labels):
        #    print(len(sen),lab)
        #print(len(sentences),len(labels))
        #print(sentences[0],labels[0])
        self.labelList = list(set(labels))
        #print(self.labelList)
        return sentences,labels
      
    def create_vacab(self, corpus):
        '''
        构造的词表
        corpus: [[word,word,...], [, , ...], ...]，
        result: store a vocab.
        '''
        #print("begin  create_vacab...")
        def word_list(corpus):
            for one in corpus:
                yield one
        vocab = learn.preprocessing.VocabularyProcessor(self._sequenceLength,0,tokenizer_fn=word_list)
        vocab.fit(corpus)
        #vocab.save(config.wordPiecePos)
        #print("end  create_vacab...")
        #vocab_list = [key for key in vocab.vocabulary_._mapping.keys()]+ ['<PAD>']
        #print(len(vocab_list))
        return [key for key in vocab.vocabulary_._mapping.keys()]
    
    def getWordEmbedding_w2c(self, words):
        word_que = []
        w2c = gensim.models.Word2Vec.load("../word2vec/w2v_model")
        vocab = []
        wordEmbedding = []
        vocab.append("<PAD>")
        vocab.append("<UNK>")
        wordEmbedding.append(np.zeros(self._embeddingSize))
        wordEmbedding.append(np.random.randn(self._embeddingSize))
        for word in words:
            try:
                vector = w2c.wv[word]
                vocab.append(word)
                wordEmbedding.append(vector)
            except:
                word_que.append(word)
                #print(word + "  不存在于词向量中")
                wordEmbedding.append(np.random.randn(self._embeddingSize))
        self.wordEmbedding = np.array(wordEmbedding)        
        return vocab, word_que
    
    def getWordEmbedding_random(self, words):
        vocab = []
        wordEmbedding = []
        vocab.append("<PAD>")
        vocab.append("<UNK>")
        wordEmbedding.append(np.zeros(self._embeddingSize))
        wordEmbedding.append(np.random.randn(self._embeddingSize))
        for word in words:
            vocab.append(word)
            wordEmbedding.append(np.random.randn(self._embeddingSize))
        self.wordEmbedding = np.array(wordEmbedding)
        return vocab
    
    def encode_x(self, input, type, voca_dict):
        x = [voca_dict.get(t, voca_dict["<UNK>"]) for t in input]
        return x
            
    def encode_y(self, y):    
        x = np.zeros(len(self.labelList))
        x[y-1] = 1
        return x
    
    def gene_sqe_fn(self, vocab, sents, labels):
        '''
            sents:原句子
            labels：原句子对应的标签
            vocab: 词表
         ''' 
        #voca = learn.preprocessing.VocabularyProcessor(self._sequenceLength,0,tokenizer_fn=word_list)
        #voca.restore(voca_filepath)
        #print('gene_sqe_fn')
        sentences, lab = [], []
        #print(len(sents),len(labels))
        token2idx = {token: idx for idx, token in enumerate(vocab)}
        #idx2token = {idx: token for idx, token in enumerate(vocab)}
        #print(idx2token.get(35))
        #print('dic_len',len(token2idx))
        #print(type(vocab.vocabulary_._mapping))
        #token2id = vocab.vocabulary_._mapping
        for sent, label in zip(sents, labels):
            sentences.append(self.encode_x(sent, "x", token2idx))
            lab.append(self.encode_y(label))
            #lab.append(label-1)
            #print((x, len(x), sent),(y, label))
        #print(len(sentences),len(lab))
        return sentences, lab
            
    def data_gen(self):
        
        
        train_data ,train_labels = self.my_readData(self.config.trainPath)
        
        test_data ,test_labels = self.my_readData(self.config.testPath)
        #print(len(train_data),train_data[0],len(train_labels),train_labels[0])
        #print(len(test_data),test_data[0],len(test_labels),test_labels[0])
        vocab = self.create_vacab(train_data+test_data)
        #print(len(vocab))
        vocab = self.getWordEmbedding_random(vocab)
        #print(len(vocab))
        # 数值序列化
        train_sqe, train_l_sqe = self.gene_sqe_fn(vocab, train_data, train_labels)
        test_sqe, test_l_sqe = self.gene_sqe_fn(vocab, test_data, test_labels)
        #print(len(train_sqe), train_sqe[0], len(train_l_sqe), train_l_sqe[0],'\n')
        #print(len(train_sqe), len(train_l_sqe))
        #print(len(test_sqe), test_sqe[0], len(test_l_sqe), test_l_sqe[0])
        self.trainReviews = np.asarray(train_sqe, dtype="int64")
        self.trainLabels = np.asarray(train_l_sqe, dtype="float32")
        self.evalReviews = np.asarray(test_sqe, dtype="int64")
        self.evalLabels = np.asarray(test_l_sqe, dtype="float32")

        

if __name__ == "__main__":
    config = Config()
    data = OtherDataSet(config)
    data.data_gen()
    print("train data shape: {}".format(data.trainReviews.shape))
    print("train label shape: {}".format(data.trainLabels.shape))
    print("eval data shape: {}".format(data.evalReviews.shape))
    '''
    with open('word_queshi.txt', 'w', encoding = 'utf-8') as f:
        for word in words:
            f.write(word+'_\n')
    '''
    
    
    