# -*- coding:utf-8 -*-
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Dropout,Reshape,merge
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model
from keras.callbacks import TensorBoard, ModelCheckpoint
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras import backend as k
import tensorflow as tf
from sklearn import metrics
import h5py
from keras.optimizers import  Adam
from keras.models import Model
from utils import ROC, load_mRNA, cv_load_mRNA,get_precision_recall_f1_acc,matthews_correlation,get_current_time,save_result,clc_cindex,binary_focal_loss,plot_acc_loss
from sklearn.model_selection import StratifiedKFold, train_test_split
import os
from model import SALMON,tensorfusion,cnn_model,fcn_model,deeptype,DeepSurv,intra_fusion,inter_model_action
import json
from model import load_data,get_cv_data,load_gege, load_patho
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve
from sklearn.metrics import roc_auc_score
import xlrd


#set gpu
os.environ["CUDA_VISIBLE_DEVICES"] = '0'


def cv_test(x,y,train_index,test_index):
  
    predict=[]  #predict label
    label=[]    #original label
    predict_score=[]  #predict scores
    
    for i in range(5):
        print('\n')
        print(i+1,'th Fold ******************')
        
        
        x_train_exp, x_test = x[train_index[i]],x[test_index[i]]
        y_train_exp, y_test = y[train_index[i]],y[test_index[i]]
        
        y_test_onehot = to_categorical(y_test)
        
        fcn = fcn_model()
        plot_model(fcn, show_shapes = True, to_file = 'fcn_model.png')
        
        log_filepath = './fcn_log/'+'cv_'+str(i)
    

        tb_cb = TensorBoard(log_dir = log_filepath, write_graph = True,write_images = 1, histogram_freq = 0)
    
        adam = Adam(lr=0.000009, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.002)
    
        loss_function = 'categorical_crossentropy'
    
        fcn.compile(loss=loss_function, optimizer=adam, metrics=['accuracy'])
        
        
#         print('models layers:',fcn.layers)
#         print('models config:',fcn.get_config())
#         print('models summary:',fcn.summary())
        
#         #get layers by name
#         layer1 = fcn.get_layer(name='hidden1')
#         layer1_W_pro = layer1.get_weights()
#         layer2 = fcn.get_layer(name='output')
#         layer2_W_pro = layer2.get_weights()

        
        x_train, x_val, y_train, y_val = train_test_split(x_train_exp,y_train_exp,test_size=0.2,stratify=y_train_exp)
        y_train_onehot = to_categorical(y_train)
        y_val_onehot = to_categorical(y_val)
        
    
        history = fcn.fit(x_train, y_train_onehot, epochs=50, batch_size=32, validation_data = (x_val, y_val_onehot), callbacks=[tb_cb])
        
        
#         #get layers1 wights
#         layer1_W_end = layer1.get_weights()
#         #layer1_W_end - layer1_W_pro
 
#         layer2_W_end = layer2.get_weights()
#         #layer2_W_end - layer2_W_pro
        
        #save weight
        fcn.save_weights('weight.h5')

        

        label.extend(y_test.astype(np.int)[:,0].tolist())  
        y_pred = fcn.predict(x_test)  
        auc = ROC(np.argmax(y_test_onehot,1), y_pred[:, 1])
        print('auc=',auc)
        predict.extend(np.argmax(y_pred,1))  
        predict_score.extend(y_pred[:,1].tolist())
        
        y_pred_train = fcn.predict(x_train) 
        auc = ROC(np.argmax(y_train_onehot,1), y_pred_train[:, 1])
        print('train auc=',auc)
        
        
#     save_result(label,predict)
   
    print('\n')
    precision,recall,f1,acc=get_precision_recall_f1_acc(label,predict)
    print('precision,recall,f1,acc= %f%f%f%f',precision,recall,f1,acc)
    cindex = clc_cindex(label, predict_score)
    print('\n')
    print('cindex=',cindex)
    auc = ROC(label, predict_score)
    print('\n')
    print('auc=',auc)
   
    return auc

def cv_validation():
    train_index = np.load('/media/user/Disk 02/wangzhiqin/TensorMulti/data/cv_data/train_index.npy')
    
    test_index = np.load('/media/user/Disk 02/wangzhiqin/TensorMulti/data/cv_data/test_index.npy')
    dataPath = '/media/user/Disk 02/wangzhiqin/TensorMulti/data/label.xls'
    data = xlrd.open_workbook(dataPath)
    table = data.sheet_by_name('sheet1')
    colum = table.ncols
    row = table.nrows
    patientID = []
    label = []
    for i in range(1,row):
        label.append(table.cell(i, 1).value)
        patientID.append(table.cell(i, 0).value)
        
    data_mRNA = load_gege()
    data_patho = load_patho()
    cancat = np.column_stack((data_mRNA,data_patho))  #(345,64)
    
# **************
# 
# **************

    predict=[]  #predict label
    ori_label=[]    #original label
    predict_score=[]  #predict scores    
    for i in range(5):
        print('\n')
        print(i+1,'th fold ######')
        
        x_train_gene, x_test_gene = np.array(data_mRNA)[train_index[i]], np.array(data_mRNA)[test_index[i]]
        y_train_gene, y_test_gene = np.array(label)[train_index[i]], np.array(label)[test_index[i]]
        
        y_test_gene_onehot = to_categorical(y_test_gene)
        
        x_train_gene, x_val_gene, y_train_gene, y_val_gene = train_test_split(x_train_gene,y_train_gene,test_size=0.2,stratify=y_train_gene)
        y_train_gene_onehot = to_categorical(y_train_gene)
        y_val_gene_onehot = to_categorical(y_val_gene)   
        
        fcn = fcn_model()
#         fcn = DeepSurv()
        plot_model(fcn, show_shapes = True, to_file = '/media/user/Disk 02/wangzhiqin/TensorMulti/result/fcn_model.png')
    
        fcn.summary()
       
        log_filepath = '/media/user/Disk 02/wangzhiqin/TensorMulti/result/model_log/' + 'cv_' +str(i)
        tb_cb = TensorBoard(log_dir = log_filepath, write_graph = True,write_images = 1, histogram_freq = 0)
        #checkpoint = ModelCheckpoint(filepath = '/media/user/Disk 02/wangzhiqin/TensorMulti/save_model',monitor='val_acc',mode='auto',save_best_only='True')
    
        adam = Adam(lr=0.00005, beta_1=0.8, beta_2=0.999, epsilon=1e-08, decay=0.01)
    
        loss_function = 'binary_crossentropy'
#         loss_function = binary_focal_loss(gamma=2, alpha=0.25)
    
        fcn.compile(loss= loss_function, optimizer=adam, metrics=['accuracy'])
        history = fcn.fit(x_train_gene, y_train_gene_onehot, epochs=11, batch_size=30, validation_data=[x_val_gene, y_val_gene_onehot], callbacks=[tb_cb])
        fcn.save_weights('/media/user/Disk 02/wangzhiqin/TensorMulti/save_model/' + 'cv_' +str(i) + '_weight.h5')
        
        #ori_label.extend(y_test_gene.astype(np.int)[:,0].tolist())
        ori_label.extend(y_test_gene)
        y_pred = fcn.predict(x_test_gene)
#         auc = roc(np.argmax(y_test_gene_onehot,1), y_pred[:, 1])
#         print('auc=',auc)
        predict.extend(np.argmax(y_pred,1))
        predict_score.extend(y_pred[:,1].tolist())
            # 绘制训练 & 验证的准确率值
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig('/media/user/Disk 02/wangzhiqin/TensorMulti/result/' + 'cv_'+str(i)+'acc.png')

        # 绘制训练 & 验证的损失值
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.savefig('/media/user/Disk 02/wangzhiqin/TensorMulti/result/'+ 'cv_'+str(i)+'loss.png')
        
    # print result
    print('\n')
  
    precision,recall,f1,acc=get_precision_recall_f1_acc(ori_label,predict)
    print('precision,recall,f1,acc= %f%f%f%f',precision,recall,f1,acc)
    cindex = clc_cindex(ori_label, predict_score)
    print('\n')
    print('cindex=',cindex)
    auc = ROC(ori_label, predict_score)
    print('\n')
    print('auc=',auc)    
    
        
        
        
        
def cnn_test(x_train, x_val, x_test, y_train, y_val, y_test):
    cnn = cnn_model()
    plot_model(cnn, show_shapes = True, to_file = 'cnn_model.png')
    
    log_filepath = '/cnn_log'
    tb_cb = TensorBoard(log_dir = log_filepath, write_graph = True,write_images = 1, histogram_freq = 0)
    
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    
    loss_function = 'mse'
    
    cnn.compile(loss=loss_function, optimizer=adam, metrics=[])
    
    history = model.fit(x_train, y_train, epochs=20, batch_size=3, validation_data=[x_val, y_val], callbacks=[tb_cb])

    
def fcn_test(x_train, x_val, x_test, y_train_onehot, y_val_onehot, y_test_onehot):
    
    fcn = fcn_model()

    plot_model(fcn, show_shapes = True, to_file = '/media/user/Disk 02/wangzhiqin/TensorMulti/result/fcn_model.png')
    
    fcn.summary()
    
    log_filepath = '/media/user/Disk 02/wangzhiqin/TensorMulti/result/fcn_log'
    tb_cb = TensorBoard(log_dir = log_filepath, write_graph = True,write_images = 1, histogram_freq = 0)
    checkpoint = ModelCheckpoint(filepath = '/media/user/Disk 02/wangzhiqin/TensorMulti/save_model',monitor='val_acc',mode='auto',save_best_only='True')
    
    adam = Adam(lr=0.00005, beta_1=0.8, beta_2=0.999, epsilon=1e-08, decay=0.01)
    
    loss_function = 'binary_crossentropy'
    
    fcn.compile(loss=loss_function, optimizer=adam, metrics=['accuracy'])
    #fcn.compile(loss=loss_function, optimizer=adam, metrics=['accuracy', auroc])
   
    history = fcn.fit(x_train, y_train_onehot, epochs=50, batch_size=30, validation_data=[x_val, y_val_onehot], callbacks=[tb_cb])
    # 绘制训练 & 验证的准确率值
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('/media/user/Disk 02/wangzhiqin/TensorMulti/result/acc.png')

    # 绘制训练 & 验证的损失值
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('/media/user/Disk 02/wangzhiqin/TensorMulti/result/loss.png')
 
    y_pred = fcn.predict(x_test)
    np.savetxt('/media/user/Disk 02/wangzhiqin/TensorMulti/result/label_predict/predict_label.txt',np.argmax(y_pred,1),fmt='%.1f')
    np.savetxt('/media/user/Disk 02/wangzhiqin/TensorMulti/result/label_predict/label.txt',np.argmax(y_test_onehot,1),fmt='%.1f')
    auc = ROC(np.argmax(y_test_onehot,1), y_pred[:, 1])
    
    print('auc=',auc)
    
def fusion_test():
    train_index = np.load('/media/user/Disk 02/wangzhiqin/TensorMulti/data/cv_data/train_index.npy')
    
    test_index = np.load('/media/user/Disk 02/wangzhiqin/TensorMulti/data/cv_data/test_index.npy')
    dataPath = '/media/user/Disk 02/wangzhiqin/TensorMulti/data/label.xls'
    data = xlrd.open_workbook(dataPath)
    table = data.sheet_by_name('sheet1')
    colum = table.ncols
    row = table.nrows
    patientID = []
    label = []
    for i in range(1,row):
        label.append(table.cell(i, 1).value)
        patientID.append(table.cell(i, 0).value)
        
    data_mRNA = load_gege()
    data_patho = load_patho()
    
    # *******
    # 用于绘制acc_loss图
    # *******
    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []
 
    predict=[]  #predict label
    ori_label=[]    #original label
    predict_score=[]  #predict scores    
    for i in range(5):
        print('\n')
        print(i+1,'th fold ######')
        
        x_train_gene, x_test_gene = np.array(data_mRNA)[train_index[i]], np.array(data_mRNA)[test_index[i]]
        x_train_patho, x_test_patho = np.array(data_patho)[train_index[i]], np.array(data_patho)[test_index[i]]
        y_train_gene, y_test_gene = np.array(label)[train_index[i]], np.array(label)[test_index[i]]

        
#         x_train_gene = np.expand_dims(x_train_gene, axis=2)
#         x_train_patho = np.expand_dims(x_train_patho, axis=2)
#         x_test_gene = np.expand_dims(x_test_gene, axis=2)
#         x_test_patho = np.expand_dims(x_test_patho, axis=2)
        
        y_test_gene_onehot = to_categorical(y_test_gene)   
        y_train_gene_onehot = to_categorical(y_train_gene) 
        
#         fusion_model = tensorfusion()
#         fusion_model = intra_tensor()
        fusion_model = intra_fusion()
    
    
        plot_model(fusion_model, show_shapes = True, to_file = '/media/user/Disk 02/wangzhiqin/TensorMulti/result/fusion_model.png')
    
        fusion_model.summary()
       
        log_filepath = '/media/user/Disk 02/wangzhiqin/TensorMulti/result/model_log/' + 'cv_' +str(i)
        tb_cb = TensorBoard(log_dir = log_filepath, write_graph = True,write_images = 1, histogram_freq = 0)
        #checkpoint = ModelCheckpoint(filepath = '/media/user/Disk 02/wangzhiqin/TensorMulti/save_model',monitor='val_acc',mode='auto',save_best_only='True')
    
        adam = Adam(lr=0.00005, beta_1=0.8, beta_2=0.999, epsilon=1e-08, decay=0.01)
    
        loss_function = 'binary_crossentropy'
#         loss_function = binary_focal_loss(gamma=2, alpha=0.24)
    
        fusion_model.compile(loss=[loss_function], optimizer=adam, metrics=['accuracy'])
        
        # ************
        # 没有加偏置，32*32
        # ************
#         history = fusion_model.fit([x_train_gene, x_train_patho], y_train_gene_onehot, epochs=48, batch_size=30,  validation_split = 0.2, callbacks=[tb_cb])
        
        # *********
        # 加偏置后，33*33
        # *********        
#         history = fusion_model.fit([x_train_gene, x_train_patho, np.ones(x_train_gene.shape[0])], y_train_gene_onehot, epochs=100, batch_size=30,  validation_split = 0.2, callbacks=[tb_cb])
        
        
        # 单模态+模态内相互作用
        history = fusion_model.fit(x_train_gene, y_train_gene_onehot, epochs=26, batch_size=30,  validation_split = 0.2, callbacks=[tb_cb])

        fusion_model.save_weights('/media/user/Disk 02/wangzhiqin/TensorMulti/save_model/' + 'cv_' +str(i) + '_weight.h5')
        fusion_model.save('/media/user/Disk 02/wangzhiqin/TensorMulti/save_model/model/'+'cv_'+str(i) + '_model.h5')
        
        #ori_label.extend(y_test_gene.astype(np.int)[:,0].tolist())
        ori_label.extend(y_test_gene)
        
        # ***********
        # 没有加偏置，32*32
        # ***********
#         y_pred = fusion_model.predict([x_test_gene,x_test_patho])
        
        # **********
        # 加偏置后，33*33
        # ***********        
#         y_pred = fusion_model.predict([x_test_gene,x_test_patho, np.ones(x_test_gene.shape[0])])
        
        
        # ********
        # 模态内相互作用
        # ********
        y_pred = fusion_model.predict(x_test_gene)
        
        
#         auc = roc(np.argmax(y_test_gene_onehot,1), y_pred[:, 1])
#         print('auc=',auc)
        predict.extend(np.argmax(y_pred,1))
        predict_score.extend(y_pred[:,1].tolist())
        
        # ********
        # 绘制acc_loss图
        # ********
        train_acc.append(history.history['acc'])
        val_acc.append(history.history['val_acc'])
        train_loss.append(history.history['loss'])
        val_loss.append(history.history['val_loss'])
        
        
    plot_acc_loss(train_acc, val_acc, train_loss, val_loss, mode =1)
    
    # save label
    np.savetxt('/media/user/Disk 02/wangzhiqin/TensorMulti/result/label_predict/predict_label.txt',predict,fmt='%.1f')
    np.savetxt('/media/user/Disk 02/wangzhiqin/TensorMulti/result/label_predict/label.txt',ori_label,fmt='%.1f')

    # print result
    print('\n')
  
    precision,recall,f1,acc=get_precision_recall_f1_acc(ori_label,predict)
    print('precision,recall,f1,acc= %f%f%f%f',precision,recall,f1,acc)
    cindex = clc_cindex(ori_label, predict_score)
    print('\n')
    print('cindex=',cindex)
    auc = ROC(ori_label, predict_score)
    print('\n')
    print('auc=',auc)  
    return precision,recall,f1,acc,auc
    
def main():
    x,y = cv_load_mRNA()
    train_index_path = 'data/train_index.txt'
    test_index_path = 'data/test_index.txt'
    filename = open(train_index_path,'r')
    train_index = filename.read()
    train_index = json.loads(train_index)
    filename = open(test_index_path,'r')
    test_index = filename.read()
    test_index = json.loads(test_index)   
    cv_test(x,y,train_index,test_index)
        
#     x_train,x_val,x_test,y_train,y_val,y_test,y_train_onehot,y_val_onehot,y_test_onehot=load_mRNA()
#     fcn_test(x_train, x_val, x_test, y_train_onehot, y_val_onehot, y_test_onehot)

    
    
    
if __name__ == '__main__':
    #train_gene, val_gene, test_gene,train_patho,val_patho,test_patho,train_label,val_label,test_label = load_data()
    #fcn_test(train_gene, val_gene, test_gene, train_label, val_label, test_label)
    #get_cv_data()
    cv_validation()
#     fusion_test()

#     Precision,Recall,F1,Acc,Auc = [],[],[],[],[]
    
#     for i in range(10):
#         precision,recall,f1,acc,auc = fusion_test()
#         Precision.append(precision)
#         Recall.append(recall)
#         F1.append(f1)
#         Acc.append(acc)
#         Auc.append(auc)
    
#     print('##########')
#     print('Precision')
#     print('\n')
#     print(Precision)
#     print('\n')
#     print('Recall')
#     print('\n')
#     print(Recall)
#     print('\n')
#     print('F1')
#     print('\n')
#     print(F1)
#     print('\n')
#     print('Acc')
#     print('\n')
#     print(Acc)
#     print('\n')
#     print('Auc')
#     print('\n') 
#     print(Auc)
#     print('\n')
    
#     print('Precision')
#     print('\n')
#     print(np.mean(Precision))
#     print('\n')
#     print(np.mean(Recall))
#     print('\n')
#     print(np.mean(F1))
#     print('\n')
#     print(np.mean(Acc))
#     print('\n')
#     print(np.mean(Auc))
            
        
    



