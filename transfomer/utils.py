# -*- coding: utf-8 -*-
import json
import numpy as np
import matplotlib.pyplot as plt

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)
        
        
def drawIndexGraph(x_data, y_data, title, x_y_axix_name, labels, graph_ex = True):
    '''
        x: x axix len(x) = len(y[i]) , type: list
        y: n*len(x) Demension data, len(y) = len(name), type: list
        title: length = len(y), type: list
        x_y_axix_name: [x_axix_name, y_axix_name], type: list
        labels: subgraph label, type:list
    '''
    store_path = "transformer/graph/"
    color = ['green', 'red', 'skyblue', 'yellow', 'blue', 'black' , 'violet', 'gray', 'pink', 'orange']
    if len(labels) > len(color): raise Exception("color categary too less, list out ")
    #sub_axix = filter(lambda x:x%x_size == 0, x_data)
    plt.title(title)
    for i in range(len(labels)):
        plt.plot(x_data, y_data[i], color=color[i], label=labels[i])
    if graph_ex:
        plt.legend() # 显示图例
    plt.xlabel(x_y_axix_name[0])
    plt.ylabel(x_y_axix_name[1])
    plt.savefig(store_path + title + '.jpg')
    plt.show()
    

def readfile(filepath, file_format = 'json'):
    print(filepath)
    if file_format == 'json':
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.loads(f.read())
        return data
    #elif file_format == 'txt':
        
    #elif file_format == 'csv':
    
    #elif file_format == 'Json':
        
    else:
        raise Exception('File format not found...')


def readData(datas, preName): 
    '''
        {
         "step": 20, 
         "loss": 12.267124891281128, 
         "accs": 0.1640625, 
         "precisions": 0.027343750000000003, 
         "recalls": 0.16666666666666666, 
         "f_betas": 0.046850804466314705
         }
    ''' 
    xs, loss, accs, pres, recs, fs = [], [], [], [], [], []   
    for data in datas:
        xs.append(data['step'])
        loss.append(data['loss'])
        accs.append(data['accs'])
        pres.append(data['precisions'])
        recs.append(data['recalls'])
        fs.append(data['f_betas'])
    if preName == 'dev':
        xs = range(len(xs))
    return xs, loss, accs, pres, recs, fs


if __name__ == "__main__":
    sourcepath = "transformer/"
    filepaths = ['2-1','2-3','2-5']
    filePreName = ['dev', 'train']
    filePostName = '_re.json'
    labels = ['t_loss', 't_accs', 't_prec', 't_recall', 't_f']
    train_xs, train_loss, train_accs, train_pres, train_recs, train_fs = [], [], [], [], [], []    
    dev_xs, dev_loss, dev_accs, dev_pres, dev_recs, dev_fs = [], [], [], [], [], []
    for preName  in filePreName :
        for filepath in filepaths:
            data = readfile(sourcepath + filepath+ '/' + preName + filePostName)
            xs, loss, accs, pres, recs, fs = readData(data, preName)
            if preName == 'dev':
                dev_xs = xs
                dev_loss.append(loss)
                dev_accs.append(accs)
                dev_pres.append(pres)
                dev_recs.append(recs)
                dev_fs.append(fs)            
            else:
                train_xs = xs
                print(len(train_xs))
                train_loss.append(loss)
                train_accs.append(accs)
                train_pres.append(pres)
                train_recs.append(recs)
                train_fs.append(fs)
        
    print('dev' , len(dev_xs), len(dev_loss))
    print('train', len(train_xs), len(train_loss))    
    for preName  in filePreName :
        if preName == 'dev':
            drawIndexGraph(dev_xs, dev_loss, 'd_loss', ['step','value'], filepaths)
            drawIndexGraph(dev_xs, dev_accs, 'd_accs', ['step','value'], filepaths)
            drawIndexGraph(dev_xs, dev_pres, 'd_prec', ['step','value'], filepaths)
            drawIndexGraph(dev_xs, dev_recs, 'd_recall', ['step','value'], filepaths)
            drawIndexGraph(dev_xs, dev_fs, 'd_f', ['step','value'], filepaths)
        else:
            drawIndexGraph(train_xs, train_loss, 't_loss', ['step','value'], filepaths)
            drawIndexGraph(train_xs, train_accs, 't_accs', ['step','value'], filepaths)
            drawIndexGraph(train_xs, train_pres, 't_prec', ['step','value'], filepaths)
            drawIndexGraph(train_xs, train_recs, 't_recall', ['step','value'], filepaths)
            drawIndexGraph(train_xs, train_fs, 't_f', ['step','value'], filepaths)
        
            
    
    



    