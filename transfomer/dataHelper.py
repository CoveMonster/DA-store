# -*- coding: utf-8 -*-

#数据预处理的类，生成训练集和测试集
from tensorflow.contrib import learn
import tensorflow as tf
import numpy as np
from Paras import Config
import gensim

class Dataset(object):
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
        print("beginning  reading...")
        sentences, labels = [], []
        with open(filepath,'r', encoding ='utf-8') as f:
            for line in f:
                new_sen = (line.split("||")[0] + line.split("||")[1]).split(" ")
                if len(new_sen)  >= self._sequenceLength: continue  
                sentences.append(new_sen + ["<PAD>"]*(self._sequenceLength - len(new_sen)))
                labels.append(int(line.split("||")[2].strip()))
        print("ending  reading...")
        self.labelList = list(set(labels))
        return sentences,labels
    
    
    def create_vacab(self, corpus):
        '''
        构造的词表
        corpus: [[word,word,...], [, , ...], ...]，
        result: store a vocab.
        '''
        print("begin  create_vacab...")
        def word_list(corpus):
            for one in corpus:
                yield one
        vocab = learn.preprocessing.VocabularyProcessor(self._sequenceLength,0,tokenizer_fn=word_list)
        vocab.fit(corpus)
        #vocab.save(config.wordPiecePos)
        print("end  create_vacab...")
        #vocab_list = [key for key in vocab.vocabulary_._mapping.keys()]+ ['<PAD>']
        #print(len(vocab_list))
        return [key for key in vocab.vocabulary_._mapping.keys()] + ['<PAD>','<UNK>']
    
    def My_getWordEmbedding(self, words):
        w2c = gensim.models.load("../word2vec/w2v_model")
        vocab = []
        wordEmbedding = []
        vocab.append("PAD")
        vocab.append("UNK")
        wordEmbedding.append(np.zeros(self._embeddingSize))
        wordEmbedding.append(np.random.randn(self._embeddingSize))
        for word in words:
            try:
                vector = w2c.wv[word]
                vocab.append(word)
                wordEmbedding.append(vector)
            except:
                print(word + "不存在于词向量中")
                
        return vocab, np.array(wordEmbedding)
    
    def encode(self, inputs, types, token2idx):
        #print('encode')
        if types == 'x':
            x = [token2idx.get(t, token2idx["<UNK>"]) for t in inputs]
        else:
            x = [0,0,0,0,0,0]
            x[input-1] = 1
        return x
    
    def gene_sqe_fn(self, vocab, sents, labels):
        '''
            sents:原句子
            labels：原句子对应的标签
            vocab: 词表
         ''' 
        #voca = learn.preprocessing.VocabularyProcessor(self._sequenceLength,0,tokenizer_fn=word_list)
        #voca.restore(voca_filepath)
        token2idx = {token: idx for idx, token in enumerate(vocab)}
        
        for sent, label in zip(sents, labels):
            x = self.encode(sent, "x", token2idx)
            y = self.encode(label, "y", token2idx)
            #print((x, len(x), sent),(y, label))
            #yield (x, len(x), sent),(y, label)
            yield x, y
    
    
    def dataGen_Word(self, sentences, labels, vocab, batch_size, shuffle=False):
        '''
        Returns
        xs: tuple of
            x: int32 tensor. (N, T1)
            x_seqlens: int32 tensor. (N,)
            sents1: str tensor. (N,)
        ys: tuple of
            y: int32 tensor. (N, T2)
            y_seqlen: int32 tensor. (N, )
            
            
            先把句子读取出来，根据语料生成词表，将语料进行词表序列化（补齐长度），把数字标签转化为向量
            最后根据训练需要进行batch复制
        '''
        '''
        shapes = (([None], (), ()),
              ([None], ()))
        types = ((tf.int32, tf.int32, tf.string),
             (tf.int32, tf.int32))
        paddings = ((0, 0, ''),
                (0, 0))
        '''
        shapes = ([None, ], [None, ])
        types = (tf.int32, tf.int32)
        paddings = (0, 0)
        dataset = tf.data.Dataset.from_generator(
                self.gene_sqe_fn,
                output_shapes=shapes,
                output_types=types,
                args=(vocab, sentences, labels))
        if shuffle: # for training
            dataset = dataset.shuffle(128*batch_size)
        dataset = dataset.repeat()  # iterate forever
        # 可能最后一个batch不足一个batch size，所以进行填充；prefetch是
        print(3)
        dataset = dataset.padded_batch(batch_size, shapes, paddings).prefetch(1)
        return dataset
    
    
    def get_batch(self, fpath, batch_size, shuffle=False):
        '''Gets training / evaluation mini-batches
        fpath: source file path. string.
        batch_size: scalar
        shuffle: boolean
    
        Returns
        batches
        num_batches: number of mini-batches
        num_samples
        '''
        #  sents1: list of source sents  sents2: list of target sents
        sentences, labels = self.my_readData(fpath)
        vocab = self.create_vacab(sentences)
        print('vocab create finished...',len(vocab))
        batches = self.dataGen_Word(sentences, labels, vocab, batch_size, shuffle)
        num_batches = self.calc_num_batches(len(sentences), batch_size)
        return batches, num_batches, len(sentences)
    
    def calc_num_batches(self, total_num, batch_size):
        '''Calculates the number of batches.
        total_num: total sample number
        batch_size
    
        Returns
        number of batches, allowing for remainders.'''
        return total_num // batch_size + int(total_num % batch_size != 0)

if __name__ == '__main__':       
    config = Config()
    data = Dataset(config)
    sentences, labels = data.my_readData(config.dataSource)
 
    #print(type(sentences),sentences[0],type(labels),labels[0])
    vocab = data.create_vacab(sentences)
    token2idx = {token: idx for idx, token in enumerate(vocab)}
    print(token2idx['<UNK>'])

    batches, num_batches, sam_num = data.get_batch(config.dataSource, 1, shuffle=False)
    '''
    iterator = batches.make_one_shot_iterator()
    one_element = iterator.get_next()
    with tf.Session() as sess:
        try:
            batch_num=0
            while True:
                one_batch = sess.run(one_element)
                print('Batch No. %d:' % batch_num)
                print(one_batch)
                print('')
                batch_num+=1
 
        except tf.errors.OutOfRangeError:
            print('end!')
    
    '''
    iter = tf.data.Iterator.from_structure(batches.output_types, batches.output_shapes)
    train_init_op = iter.make_initializer(batches)
    with tf.Session() as sess:
        sess.run(train_init_op)
        try:
            xs, ys = iter.get_next()
            xs = np.array(xs)
            print(type(xs),xs.shape,xs.size)
        except tf.errors.OutOfRangeError:
            print("over")
    

    
