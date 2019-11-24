# -*- coding: utf-8 -*-
from dataH import OtherDataSet
from Paras import Config
from analysis import fixedPositionEmbedding, nextBatch, get_binary_metrics, get_multi_metrics, mean
import tensorflow as tf
from model import Transformer
import os,datetime
import json
from utils import MyEncoder
#训练模型

# 实例化配置参数对象
config = Config()
# 生成训练集和验证集
data = OtherDataSet(config)
data.data_gen()
trainReviews = data.trainReviews # 训练集
trainLabels = data.trainLabels # 训练标签
evalReviews = data.evalReviews
evalLabels = data.evalLabels
print(trainReviews.shape, trainLabels.shape, evalReviews.shape, evalLabels.shape)
wordEmbedding = data.wordEmbedding
labelList = data.labelList
print(wordEmbedding.shape, len(labelList))
embeddedPosition = fixedPositionEmbedding(config.batchSize, config.sequenceLength)

# 定义计算图
with tf.Graph().as_default():

    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_conf.gpu_options.allow_growth=True
    session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9  # 配置gpu占用率  

    sess = tf.Session(config=session_conf)
    train_result = []
    dev_result = []
    # 定义会话
    with sess.as_default():
        transformer = Transformer(config, wordEmbedding)
        
        globalStep = tf.Variable(0, name="globalStep", trainable=False)
        # 定义优化函数，传入学习速率参数
        optimizer = tf.train.AdamOptimizer(config.training.learningRate)
        # 计算梯度,得到梯度和变量
        gradsAndVars = optimizer.compute_gradients(transformer.loss)
        # 将梯度应用到变量下，生成训练器
        trainOp = optimizer.apply_gradients(gradsAndVars, global_step=globalStep)
        
        # 用summary绘制tensorBoard
        gradSummaries = []
        for g, v in gradsAndVars:
            if g is not None:
                tf.summary.histogram("{}/grad/hist".format(v.name), g)
                tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
        
        outDir = os.path.abspath(os.path.join(os.path.curdir, config.summarySavePath))
        print("Writing to {}\n".format(outDir))
        
        lossSummary = tf.summary.scalar("loss", transformer.loss)
        summaryOp = tf.summary.merge_all()
        
        trainSummaryDir = os.path.join(outDir, "train")
        trainSummaryWriter = tf.summary.FileWriter(trainSummaryDir, sess.graph)
        
        evalSummaryDir = os.path.join(outDir, "eval")
        evalSummaryWriter = tf.summary.FileWriter(evalSummaryDir, sess.graph)
        
        
        # 初始化所有变量
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        
        # 保存模型的一种方式，保存为pb文件
        savedModelPath = config.savedModelPath
        if os.path.exists(savedModelPath):
            os.rmdir(savedModelPath)
        builder = tf.saved_model.builder.SavedModelBuilder(savedModelPath)
            
        sess.run(tf.global_variables_initializer())

        def trainStep(batchX, batchY):
            """
            训练函数
            """   
            feed_dict = {
              transformer.inputX: batchX,
              transformer.inputY: batchY,
              transformer.dropoutKeepProb: config.model.dropoutKeepProb,
              transformer.embeddedPosition: embeddedPosition
            }
            _, summary, step, loss, predictions = sess.run(
                [trainOp, summaryOp, globalStep, transformer.loss, transformer.predictions],
                feed_dict)
            print('pre_type:',type(predictions), predictions.shape ,'y_type:',type(tf.argmax(batchY, 1).eval()), tf.argmax(batchY, 1).shape) 
            if config.numClasses == 1:
                acc, recall, prec, f_beta = get_binary_metrics(pred_y=predictions, true_y=batchY) 
            elif config.numClasses > 1:
                acc, recall, prec, f_beta = get_multi_metrics(pred_y=predictions, true_y=tf.argmax(batchY, 1).eval(),
                                                              labels=labelList)
               
            trainSummaryWriter.add_summary(summary, step)
            
            return loss, acc, prec, recall, f_beta

        def devStep(batchX, batchY):
            """
            验证函数
            """
            feed_dict = {
              transformer.inputX: batchX,
              transformer.inputY: batchY,
              transformer.dropoutKeepProb: 1.0,
              transformer.embeddedPosition: embeddedPosition
            }
            summary, step, loss, predictions = sess.run(
                [summaryOp, globalStep, transformer.loss, transformer.predictions],
                feed_dict)
            
            if config.numClasses == 1:
                acc, recall, prec, f_beta = get_binary_metrics(pred_y=predictions, true_y=batchY)

                
            elif config.numClasses > 1:
                acc, recall, prec, f_beta = get_multi_metrics(pred_y=predictions, true_y=tf.argmax(batchY, 1).eval(),
                                                              labels=labelList)
                
            evalSummaryWriter.add_summary(summary, step)
            
            return loss, acc, prec, recall, f_beta
        
        for i in range(config.training.epoches):
            # 训练模型
            print("start training model")
            for batchTrain in nextBatch(trainReviews, trainLabels, config.batchSize):
                print(batchTrain[0].shape, batchTrain[1].shape)
                loss, acc, prec, recall, f_beta = trainStep(batchTrain[0], batchTrain[1])
                
                currentStep = tf.train.global_step(sess, globalStep) 
                train_result.append({"step":currentStep, "loss":loss, "accs":acc, "precisions":prec, "recalls":recall, "f_betas":f_beta})
                print("train: step: {}, loss: {}, acc: {}, recall: {}, precision: {}, f_beta: {}".format(
                    currentStep, loss, acc, recall, prec, f_beta))
                if currentStep % config.training.evaluateEvery == 0:
                    print("\nEvaluation:")
                    
                    losses = []
                    accs = []
                    f_betas = []
                    precisions = []
                    recalls = []
                    
                    for batchEval in nextBatch(evalReviews, evalLabels, config.batchSize):
                        loss, acc, precision, recall, f_beta = devStep(batchEval[0], batchEval[1])
                        losses.append(loss)
                        accs.append(acc)
                        f_betas.append(f_beta)
                        precisions.append(precision)
                        recalls.append(recall)
                        
                    time_str = datetime.datetime.now().isoformat()
                    dev_result.append({"step":currentStep, "loss":mean(losses), "accs":mean(accs), "precisions":mean(precisions),"recalls":mean(recalls), "f_betas":mean(f_betas)})
                    print("{}, step: {}, loss: {}, acc: {},precision: {}, recall: {}, f_beta: {}".format(time_str, currentStep, mean(losses), 
                                                                                                       mean(accs), mean(precisions),
                                                                                                       mean(recalls), mean(f_betas)))
                    
                if currentStep % config.training.checkpointEvery == 0:
                    # 保存模型的另一种方法，保存checkpoint文件
                    path = saver.save(sess, config.checkpointSavePath, global_step=currentStep)
                    print("Saved model checkpoint to {}\n".format(path))
                    
        inputs = {"inputX": tf.saved_model.utils.build_tensor_info(transformer.inputX),
                  "keepProb": tf.saved_model.utils.build_tensor_info(transformer.dropoutKeepProb)}

        outputs = {"predictions": tf.saved_model.utils.build_tensor_info(transformer.predictions)}

        prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs, outputs=outputs,
                                                                                      method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
        legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
                                            signature_def_map={"predict": prediction_signature}, legacy_init_op=legacy_init_op)

        builder.save()
    with open(config.train_result_index, 'w', encoding = 'utf-8') as f:
        f.write(json.dumps(train_result, cls=MyEncoder))
    with open(config.dev_result_index, 'w', encoding = 'utf-8') as f:
        f.write(json.dumps(dev_result , cls=MyEncoder))
    print('验证结果已保存...')
    