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
from model import SALMON,tensorfusion,cnn_model,fcn_model,deeptype,DeepSurv,intra_fusion
import json
from model import load_data,get_cv_data,load_gege, load_patho
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve
from sklearn.metrics import roc_auc_score
import xlrd

#set gpu
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def cv_validation():
    train_index = np.load('/media/user/Disk 02/wangzhiqin/TensorMulti/data/cv_data/train_index.npy')
    
    test_index = np.load('/media/user/Disk 02/wangzhiqin/TensorMulti/data/cv_data/test_index.npy')
    label = get_label()
    os_time = get_os_time()
    os_state = get_state()
        
    data_patho = load_patho()
  
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
        y_test_gene_os_time = np.array(os_time)[test_index[i]]
        y_test_gene_os_state = np.array(os_state)[test_index[i]]        
        y_train_patho_onehot = to_categorical(y_train_patho)
        fcn = fcn_model()
        plot_model(fcn, show_shapes = True, to_file = '/media/user/Disk 02/wangzhiqin/TensorMulti/result/patho-only/fcn_model.png')
    
        fcn.summary()
       
        log_filepath = '/media/user/Disk 02/wangzhiqin/TensorMulti/result/patho-only/model_log/' + 'cv_' +str(i)
        tb_cb = TensorBoard(log_dir = log_filepath, write_graph = True,write_images = 1, histogram_freq = 0)
        #checkpoint = ModelCheckpoint(filepath = '/media/user/Disk 02/wangzhiqin/TensorMulti/save_model',monitor='val_acc',mode='auto',save_best_only='True')
    
        adam = Adam(lr=0.00005, beta_1=0.8, beta_2=0.999, epsilon=1e-08, decay=0.01)
    
        loss_function = 'binary_crossentropy'
#         loss_function = binary_focal_loss(gamma=2, alpha=0.25)
    
        fcn.compile(loss= loss_function, optimizer=adam, metrics=['accuracy'])
        history = fcn.fit(x_train_patho, y_train_patho_onehot, epochs=11, batch_size=35, validation_split = 0.2, callbacks=[tb_cb])
        fcn.save_weights('/media/user/Disk 02/wangzhiqin/TensorMulti/result/patho-only/save_model/' + 'cv_' +str(i) + '_weight.h5')
        
        #ori_label.extend(y_test_gene.astype(np.int)[:,0].tolist())
        ori_label.extend(y_test_patho)
        ori_os_time.extend(y_test_gene_os_time)
        ori_os_state.extend(y_test_gene_os_state)        
        y_pred = fcn.predict(x_test_patho)
#         auc = roc(np.argmax(y_test_gene_onehot,1), y_pred[:, 1])
#         print('auc=',auc)
        predict.extend(np.argmax(y_pred,1))
        predict_score.extend(y_pred[:,1].tolist())
            # 绘制训练 & 验证的准确率值
#         plt.plot(history.history['acc'])
#         plt.plot(history.history['val_acc'])
#         plt.title('Model accuracy')
#         plt.ylabel('Accuracy')
#         plt.xlabel('Epoch')
#         plt.legend(['Train', 'Validation'], loc='upper left')
#         plt.savefig('/media/user/Disk 02/wangzhiqin/TensorMulti/result/patho-only/' + 'cv_'+str(i)+'acc.png')

        # 绘制训练 & 验证的损失值
#         plt.plot(history.history['loss'])
#         plt.plot(history.history['val_loss'])
#         plt.title('Model loss')
#         plt.ylabel('Loss')
#         plt.xlabel('Epoch')
#         plt.legend(['Train', 'Validation'], loc='upper left')
#         plt.savefig('/media/user/Disk 02/wangzhiqin/TensorMulti/result/patho-only/'+ 'cv_'+str(i)+'loss.png')
        
    
    # save for KM
    np.save('/media/user/Disk 02/wangzhiqin/TensorMulti/result/patho-only/km/patho_only_predict_label.npy',predict)
    np.save('/media/user/Disk 02/wangzhiqin/TensorMulti/result/patho-only/km/patho_only_os_time.npy', ori_os_time)  
    np.save('/media/user/Disk 02/wangzhiqin/TensorMulti/result/patho-only/km/patho_only_state.npy', ori_os_state)
    
    
    # print result
    print('\n')
  
    precision,recall,f1,acc=get_precision_recall_f1_acc(ori_label,predict)
    print('precision,recall,f1,acc= %f%f%f%f',precision,recall,f1,acc)
    cindex = clc_cindex(np.array(ori_os_time), -np.array(predict_score),np.array(ori_os_state))
    print('\n')
    print('cindex=',cindex)
#     np.save('/media/user/Disk 02/wangzhiqin/TensorMulti/result/patho-only/roc/patho_only_ori_label.npy',ori_label)
#     np.save('/media/user/Disk 02/wangzhiqin/TensorMulti/result/patho-only/roc/patho_only_predict_score.npy', predict_score)
    auc = ROC(ori_label, predict_score)
    print('\n')
    print('auc=',auc) 

if __name__ == '__main__':
    cv_validation()