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
from model import SALMON,tensorfusion,cnn_model,fcn_model,DeepSurv, concat_model,corr_loss
import json
from model import load_data,get_cv_data,load_gege, load_patho
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve
from sklearn.metrics import roc_auc_score
import xlrd

#set gpu
os.environ["CUDA_VISIBLE_DEVICES"] = '2'


# def precision(y_true, y_pred):
#     # Calculates the precision
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     return precision

# def recall(y_true, y_pred):
#     # Calculates the recall
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall = true_positives / (possible_positives + K.epsilon())
#     return recall

# def fbeta_score(y_true, y_pred, beta=1):
#     # Calculates the F score, the weighted harmonic mean of precision and recall.
 
#     if beta < 0:
#         raise ValueError('The lowest choosable beta is zero (only precision).')
        
#     # If there are no true positives, fix the F score at 0 like sklearn.
#     if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
#         return 0
 
#     p = precision(y_true, y_pred)
#     r = recall(y_true, y_pred)
#     bb = beta ** 2
#     fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
#     return fbeta_score
 
# def fmeasure(y_true, y_pred):
#     # Calculates the f-measure, the harmonic mean of precision and recall.
#     return fbeta_score(y_true, y_pred, beta=1)

# model.compile( 
#         optimizer=Adam(), 
#         loss='binary_crossentropy',
#         metrics = ['accuracy',  fmeasure, recall, precision])





def cv_validation():
    train_index = np.load('/media/user/Disk 02/wangzhiqin/TensorMulti/data/cv_data/train_index.npy')
    
    test_index = np.load('/media/user/Disk 02/wangzhiqin/TensorMulti/data/cv_data/test_index.npy')
    label = get_label()
    os_time = get_os_time()
    os_state = get_state()
        
    data_mRNA = load_gege()
    data_patho = load_patho()
    cancat = np.column_stack((data_mRNA,data_patho))  #(345,64)

    predict=[]  #predict label
    ori_label=[]    #original label
    ori_os_time = []
    ori_os_state = []    
    predict_score=[]  #predict scores    
    for i in range(5):
        print('\n')
        print(i+1,'th fold ######')
        
        x_train, x_test = np.array(cancat)[train_index[i]], np.array(cancat)[test_index[i]]
        y_train, y_test = np.array(label)[train_index[i]], np.array(label)[test_index[i]]
        
        y_test_onehot = to_categorical(y_test)
        y_train_onehot = to_categorical(y_train)
        y_test_gene_os_time = np.array(os_time)[test_index[i]]
        y_test_gene_os_state = np.array(os_state)[test_index[i]]        
        
 
        
        fcn = concat_model()
#         fcn = DeepSurv()
        plot_model(fcn, show_shapes = True, to_file = '/media/user/Disk 02/wangzhiqin/TensorMulti/result/concatenate/fcn_model.png')
    
        fcn.summary()
       
        log_filepath = '/media/user/Disk 02/wangzhiqin/TensorMulti/result/concatenate/model_log/' + 'cv_' +str(i)
        tb_cb = TensorBoard(log_dir = log_filepath, write_graph = True,write_images = 1, histogram_freq = 0)
        #checkpoint = ModelCheckpoint(filepath = '/media/user/Disk 02/wangzhiqin/TensorMulti/save_model',monitor='val_acc',mode='auto',save_best_only='True')
    
#         adam = Adam(lr=0.00005, beta_1=0.8, beta_2=0.999, epsilon=1e-08, decay=0.01)
        
        adam = Adam(lr=0.0005, beta_1=0.8, beta_2=0.999, epsilon=1e-08, decay=0.01)
        loss_function = 'binary_crossentropy'
#         loss_function = binary_focal_loss(gamma=2, alpha=0.25)
        fcn.compile(loss= loss_function, optimizer=adam, metrics=['accuracy'])
        
    
#         fcn.compile(loss= corr_loss, optimizer=adam, metrics=['accuracy'])
        history = fcn.fit(x_train, y_train_onehot, epochs=50, batch_size=16, validation_split = 0.2, callbacks=[tb_cb])
#         fcn.save_weights('/media/user/Disk 02/wangzhiqin/TensorMulti/result/concatenate/save_model/' + 'cv_' +str(i) + '_weight.h5')
        
        #ori_label.extend(y_test_gene.astype(np.int)[:,0].tolist())
        ori_label.extend(y_test)
        ori_os_time.extend(y_test_gene_os_time)
        ori_os_state.extend(y_test_gene_os_state)        
        y_pred = fcn.predict(x_test)
#         auc = roc(np.argmax(y_test_gene_onehot,1), y_pred[:, 1])
#         print('auc=',auc)
#         predict.extend(np.argmax(y_pred,1))
        predict.extend(np.argmax(y_pred[:,0:2],1))
        predict_score.extend(y_pred[:,1].tolist())


        
        
        
        
            # 绘制训练 & 验证的准确率值
#         plt.plot(history.history['acc'])
#         plt.plot(history.history['val_acc'])
#         plt.title('Model accuracy')
#         plt.ylabel('Accuracy')
#         plt.xlabel('Epoch')
#         plt.legend(['Train', 'Validation'], loc='upper left')
#         plt.savefig('/media/user/Disk 02/wangzhiqin/TensorMulti/result/concatenate/' + 'cv_'+str(i)+ '_' + 'acc.png')

        # 绘制训练 & 验证的损失值
#         plt.plot(history.history['loss'])
#         plt.plot(history.history['val_loss'])
#         plt.title('Model loss')
#         plt.ylabel('Loss')
#         plt.xlabel('Epoch')
#         plt.legend(['Train', 'Validation'], loc='upper left')
#         plt.savefig('/media/user/Disk 02/wangzhiqin/TensorMulti/result/concatenate/'+ 'cv_'+str(i)+ '_' + 'loss.png')


    # save for KM
#     np.save('/media/user/Disk 02/wangzhiqin/TensorMulti/result/concatenate/km/concatenate_predict_label.npy',predict)
#     np.save('/media/user/Disk 02/wangzhiqin/TensorMulti/result/concatenate/km/concatenate_os_time.npy', ori_os_time)  
#     np.save('/media/user/Disk 02/wangzhiqin/TensorMulti/result/concatenate/km/concatenate_state.npy', ori_os_state)

    # print result
    print('\n')
  
    precision,recall,f1,acc=get_precision_recall_f1_acc(ori_label,predict)
    print('precision,recall,f1,acc= %f%f%f%f',precision,recall,f1,acc)
    cindex = clc_cindex(np.array(ori_os_time), -np.array(predict_score),np.array(ori_os_state))
    print('\n')
    print('cindex=',cindex)
#     np.save('/media/user/Disk 02/wangzhiqin/TensorMulti/result/concatenate/roc/cancatenate_ori_label.npy',ori_label)
#     np.save('/media/user/Disk 02/wangzhiqin/TensorMulti/result/concatenate/roc/cancatenate_predict_score.npy', predict_score)
    auc = ROC(ori_label, predict_score)
    print('\n')
    print('auc=',auc)  

if __name__ == '__main__':
    cv_validation()