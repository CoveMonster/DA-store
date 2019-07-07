from gensim.models import word2vec
import os


# 训练词向量
def train_w2v(path):
    sourcepath = os.path.abspath('data/corpus.txt')
    sentences = word2vec.Text8Corpus(sourcepath)  # 加载语料
    model = word2vec.Word2Vec(sentences, sg=1, size=100, window=5, min_count=5,
                              negative=3, sample=0.001, hs=1, workers=4)
    model.save(path)


if __name__ == '__main__':
    train_w2v('/word2v.model')







