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
from model import SALMON,tensorfusion,cnn_model,fcn_model,DeepSurv,intra_fusion
import json
from model import load_data,get_cv_data,load_gege, load_patho
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve
from sklearn.metrics import roc_auc_score
import xlrd


#set gpu
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

def fusion_test():
    train_index = np.load('/media/user/Disk 02/wangzhiqin/TensorMulti/data/cv_data/train_index.npy')
    
    test_index = np.load('/media/user/Disk 02/wangzhiqin/TensorMulti/data/cv_data/test_index.npy')
    label = get_label()
    os_time = get_os_time()
    os_state = get_state()
        
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
    ori_os_time = []
    ori_os_state = []    
    predict_score=[]  #predict scores    
    for i in range(5):
        print('\n')
        print(i+1,'th fold ######')
        
        x_train_patho, x_test_patho = np.array(data_patho)[train_index[i]], np.array(data_patho)[test_index[i]]
        y_train_patho, y_test_patho = np.array(label)[train_index[i]], np.array(label)[test_index[i]]

        
        y_test_patho_onehot = to_categorical(y_test_patho)   
        y_train_patho_onehot = to_categorical(y_train_patho)
        
        y_test_gene_os_time = np.array(os_time)[test_index[i]]
        y_test_gene_os_state = np.array(os_state)[test_index[i]]        
        
        fusion_model = intra_fusion()
    
    
        plot_model(fusion_model, show_shapes = True, to_file = '/media/user/Disk 02/wangzhiqin/TensorMulti/result/intra_patho/fusion_model.png')
    
        fusion_model.summary()
       
        log_filepath = '/media/user/Disk 02/wangzhiqin/TensorMulti/result/intra_patho/model_log/' + 'cv_' +str(i)
        tb_cb = TensorBoard(log_dir = log_filepath, write_graph = True,write_images = 1, histogram_freq = 0)
        # 保存最优模型
    
        adam = Adam(lr=0.0004, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001)
    
        loss_function = 'binary_crossentropy'
#         loss_function = binary_focal_loss(gamma=2, alpha=0.24)
    
        fusion_model.compile(loss=[loss_function], optimizer=adam, metrics=['accuracy'])
        
        
        # 单模态+模态内相互作用
        history = fusion_model.fit(x_train_patho, y_train_patho_onehot, epochs=50, batch_size=16,  validation_split = 0.2, callbacks=[tb_cb])

#         fusion_model.save_weights('/media/user/Disk 02/wangzhiqin/TensorMulti/result/intra_patho/save_model/' + 'cv_' +str(i) + '_weight.h5')
#         fusion_model.save('/media/user/Disk 02/wangzhiqin/TensorMulti/result/intra_patho/save_model/model/'+'cv_'+str(i) + '_model.h5')

        tsne_path = '/media/user/Disk 02/wangzhiqin/TensorMulti/comparison_result/tsne/tsne_intra_patho/'
        dense4_layer_model = Model(inputs=fusion_model.input,outputs=fusion_model.get_layer('FC_4').output) 
        dense4_output = dense4_layer_model.predict(np.array(data_patho))
        np.save(tsne_path+'cv_'+str(i)+'FC_4.npy',dense4_output)
        np.save(tsne_path+'cv_'+str(i)+'lable.npy',np.array(label))


        ori_label.extend(y_test_patho)
        ori_os_time.extend(y_test_gene_os_time)
        ori_os_state.extend(y_test_gene_os_state)        
        
        
        # ********
        # 模态内相互作用
        # ********
        y_pred = fusion_model.predict(x_test_patho)
        
        predict.extend(np.argmax(y_pred,1))
        predict_score.extend(y_pred[:,1].tolist())
        
        # ********
        # 绘制acc_loss图
        # ********
#         train_acc.append(history.history['acc'])
#         val_acc.append(history.history['val_acc'])
#         train_loss.append(history.history['loss'])
#         val_loss.append(history.history['val_loss'])
        
        
#     plot_acc_loss(train_acc, val_acc, train_loss, val_loss, mode =1)
    
    # save label
#     np.save('/media/user/Disk 02/wangzhiqin/TensorMulti/result/intra_patho/roc/cindex_7/intra_patho_ori_label.npy',ori_label)
#     np.save('/media/user/Disk 02/wangzhiqin/TensorMulti/result/intra_patho/roc/cindex_7/intra_patho_predict_score.npy', predict_score)    


    # save for KM
#     np.save('/media/user/Disk 02/wangzhiqin/TensorMulti/result/intra_patho/km/intra_patho_predict_label.npy',predict)
#     np.save('/media/user/Disk 02/wangzhiqin/TensorMulti/result/intra_patho/km/intra_patho_os_time.npy', ori_os_time)  
#     np.save('/media/user/Disk 02/wangzhiqin/TensorMulti/result/intra_patho/km/intra_patho_state.npy', ori_os_state)


    # print result
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

if __name__ == '__main__':
    precision,recall,f1,acc,auc,cindex = fusion_test()
#     cindex = 0
#     while(cindex<0.583):
#         precision,recall,f1,acc,auc,cindex = fusion_test()
        
    
    

