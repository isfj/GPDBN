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
from utils import ROC, load_mRNA, cv_load_mRNA,get_precision_recall_f1_acc,matthews_correlation,get_current_time,save_result,clc_cindex,binary_focal_loss,plot_acc_loss,get_label,get_os_time, get_state
from sklearn.model_selection import StratifiedKFold, train_test_split
import os
from model import deepcca,corr_loss, deepcorrsurv
import json
from model import load_data,get_cv_data,load_gege, load_patho
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve
from sklearn.metrics import roc_auc_score
import xlrd

os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax(memory_gpu))
os.system('rm tmp')


#set gpu
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'


def pre_train():
    path = '/media/user/Disk 02/wangzhiqin/TensorMulti/data/cv_data/2020_6_23/'
    train_index = np.load(path + 'train_index.npy')
    test_index = np.load(path + 'test_index.npy')
    label = get_label()
    os_time = get_os_time()
    os_state = get_state()
        
    data_mRNA = load_gege()
    data_patho = load_patho()
    predict=[]  #predict label
    ori_label=[]    #original label
    ori_os_time = []
    ori_os_state = []    
    predict_score=[]  #predict scores    
    for i in range(5):
        print('\n')
        print(i+1,'th fold ######')
        
        x_train_gene, x_test_gene = np.array(data_mRNA)[train_index[i]], np.array(data_mRNA)[test_index[i]]
        
        x_train_patho, x_test_patho = np.array(data_patho)[train_index[i]], np.array(data_patho)[test_index[i]]
        y_train_gene, y_test_gene = np.array(label)[train_index[i]], np.array(label)[test_index[i]]
        

        
        y_test_gene_onehot = to_categorical(y_test_gene)   
        y_train_gene_onehot = to_categorical(y_train_gene) 
       
        y_test_gene_os_time = np.array(os_time)[test_index[i]]
        y_test_gene_os_state = np.array(os_state)[test_index[i]] 
        
        
        
        fusion_model = deepcorrsurv()
#         fusion_model = deepcca()
       
        
        save_path = '/media/user/Disk 02/wangzhiqin/TensorMulti/comparison_result/deepcorrsurv/'
        plot_model(fusion_model, show_shapes = True, to_file =save_path + 'deepcorrxurv.png')
        fusion_model.summary()
        log_filepath = '/media/user/Disk 02/wangzhiqin/TensorMulti/comparison_result/deepcorrsurv/model_log/' + 'cv_' +str(i)
        tb_cb = TensorBoard(log_dir = log_filepath, write_graph = True,write_images = 1, histogram_freq = 0)
        checkpoint = ModelCheckpoint(filepath = '/media/user/Disk 02/wangzhiqin/TensorMulti/comparison_result/deepcorrsurv/save_model/model/'+'cv_' + str(i)+'/weights_{epoch:02d}.h5',monitor='val_acc',mode='max',save_best_only='True',period=1)
        cbks = [tb_cb, checkpoint]
        adam = Adam(lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001)
        
        losses = {'input_cca': corr_loss, 'output': 'binary_crossentropy'}
        fusion_model.compile(loss=losses, optimizer=adam, metrics=['accuracy'])
        history = fusion_model.fit([x_train_gene, x_train_patho], [y_train_gene_onehot,y_train_gene_onehot], epochs=100, batch_size=16,  validation_split = 0.2, callbacks=cbks)
        
        
#         losses = 'binary_crossentropy'
#         fusion_model.compile(loss=losses, optimizer=adam, metrics=['accuracy'])
#         history = fusion_model.fit([x_train_gene, x_train_patho], y_train_gene_onehot, epochs=150, batch_size=16,  validation_split = 0.2, callbacks=cbks)
        
        
        
        
        ori_label.extend(y_test_gene)
        ori_os_time.extend(y_test_gene_os_time)
        ori_os_state.extend(y_test_gene_os_state) 
        y_pred = fusion_model.predict([x_test_gene,x_test_patho])

        predict.extend(np.argmax(y_pred[1],1))
        predict_score.extend(y_pred[1][:,1].tolist())

#         predict.extend(np.argmax(y_pred,1))
#         predict_score.extend(y_pred[:,1].tolist())

    # save label
    np.save('/media/user/Disk 02/wangzhiqin/TensorMulti/comparison_result/deepcorrsurv/roc/deepcorrsurv_ori_label.npy',ori_label)
    np.save('/media/user/Disk 02/wangzhiqin/TensorMulti/comparison_result/deepcorrsurv/roc/deepcorrsurv_predict_score.npy', predict_score)    

    # save for KM
    np.save('/media/user/Disk 02/wangzhiqin/TensorMulti/comparison_result/deepcorrsurv/km/deepcorrsurv_predict_label.npy',predict)
    np.save('/media/user/Disk 02/wangzhiqin/TensorMulti/comparison_result/deepcorrsurv/km/deepcorrsurv_os_time.npy', ori_os_time)  
    np.save('/media/user/Disk 02/wangzhiqin/TensorMulti/comparison_result/deepcorrsurv/km/deepcorrsurv_state.npy', ori_os_state)


    print('\n')
  
    precision,recall,f1,acc=get_precision_recall_f1_acc(ori_label,predict)
    print('precision,recall,f1,acc= %f%f%f%f',precision,recall,f1,acc)
    cindex = clc_cindex(np.array(ori_os_time), -np.array(predict_score),np.array(ori_os_state))
    print('\n')
    print('cindex=',cindex)
    auc = ROC(ori_label, predict_score)
    print('\n')
    print('auc=',auc)  
    return precision,recall,f1,acc,auc,cindex        
        
        
        
    
# def fine_tuning():
    



if __name__ == '__main__':
    pre_train()