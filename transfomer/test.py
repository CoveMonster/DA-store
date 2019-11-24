# -*- coding: utf-8 -*-

import numpy as np
import json
#vocal = learn.preprocessing.VocabularyProcessor(max_document_length,
#                                                min_frequency=0,vocabulary=None,tokenizer_fn=None)
    
def create_set():
    file_path_list = ["..//data//compare.txt","..//data//condition.txt","..//data//expansion.txt",
                      "..//data/parallel.txt","..//data//result.txt","..//data//time.txt"]
    train = []
    test = []
    for filepath in file_path_list:
        list = readData(filepath)
        print(len(list))
        train += list[0:399]
        test += list[400:499]
    with open('train.txt', 'w', encoding = 'utf-8')as f1:
        for line in train:
            f1.write(line)
        
    with open('test.txt', 'w' , encoding = 'utf-8') as f2:
        for line in test:
            f2.write(line)
          
def readData(filepath):
    list = []
    with open(filepath,'r',encoding = 'utf-8') as f:
        for line in f.readlines():
            if len(list)>=500: continue
            sentence = (line.split("||")[0] + line.split("||")[1]).split(" ")
            if len(sentence) >= 128 :continue
            list.append(line)
    return list

def countData(filepath):
    list = []
    count = 0
    with open(filepath,'r',encoding = 'utf-8') as f:
        for line in f.readlines():
            sentence = []
            if len(list)>=500: continue
            for i in (line.split("||")[0] + line.split("||")[1]).split(" "):
                if i == '':continue
                sentence.append(i)
            if len(sentence) >= 128 or len(sentence) <= 0 :continue
            count+=1
    return count

def countCor():
    '''
        result:
            compare 2409
            condition 672
            expansion 14359
            parallel 2282
            result 2521
            time 851
            
    '''
    sourcepath = "..//data//"
    file_path_list = ["compare.txt","condition.txt","expansion.txt",
                      "parallel.txt","result.txt","time.txt"]
    for filepath in file_path_list:
       print(filepath.split('.')[0], countData(sourcepath + filepath))
       
if __name__ == "__main__":
    countCor()
    