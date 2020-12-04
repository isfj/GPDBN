# -*- coding:utf-8 -*-
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc

from sklearn.model_selection import StratifiedKFold
import h5py
import numpy as np
from keras.utils.np_utils import to_categorical
from keras import backend as K
import tensorflow as tf
from compiler.ast import flatten
import time
from lifelines.utils import concordance_index
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import xlrd
import xlwt


'''
class rocHistory(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = {}):
        y_pred = self.model.predict(self.validation_data[0:32])
        yp = []
        for i in xrange(0, len(y_pred)):
            yp.append(y_pred[i][0])
        
        yt = []
        for x in self.validation_data[2]:
            yt.append(x[0])
        
        auc = roc_auc_score(yt, yp)
        self.aucs.append(auc)
        print('val-auc:', auc)
        print('\n')
        return
'''
def clc_cindex(label,predict,censor):
    cindex = concordance_index(label,predict,censor)
#     print('cindex=',cindex)
    return cindex




def ROC(label,predict):
    fpr, tpr, threshold = roc_curve(label, predict)
    sp=1-fpr
#     np.savetxt('/media/user/Disk 02/wangzhiqin/TensorMulti/result/roc/sp.txt',sp,fmt='%.4f')
#     np.savetxt('/media/user/Disk 02/wangzhiqin/TensorMulti/result/roc/fpr.txt',fpr,fmt='%.4f')
#     np.savetxt('/media/user/Disk 02/wangzhiqin/TensorMulti/result/roc/tpr.txt',tpr,fmt='%.4f')
#     np.savetxt('/media/user/Disk 02/wangzhiqin/TensorMulti/result/roc/threshold.txt',threshold,fmt='%.4f')
    AUC = auc(fpr, tpr)
    
#     plt.figure()
#     plt.plot(fpr, tpr, color="red", label='ROC curve(auc = %0.2f)' % AUC)
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.0])
#     plt.xlabel("FPR")
#     plt.ylabel("TPR")
#     plt.title('TCGA-BRC-ROC')
#     plt.legend(loc='lower right')
#     plt.savefig("/media/user/Disk 02/wangzhiqin/TensorMulti/result/roc/roc.png")
   
    return AUC

def get_sp(label,predict):
    fpr, tpr, threshold = roc_curve(label, predict)
    sp=1-fpr
    

def get_current_time():
    return time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))

def save_result(original_label,predict,file_="label_result"):
    curr_t = get_current_time()
    fiw1 = open('results/' + file_ + '_' + curr_t + '_original_label.txt', 'w')
    fiw2 = open('results/' + file_ + '_' + curr_t + '_predict.txt', 'w')
    for s in original_label:
        for i in s:
            fiw1.write(str(i)+'')
        fiw1.write('\n')
    fiw1.close()
    for s in predict:
        for i in s:
            fiw2.write(str(i)+'')
        fiw2.write('\n')
    fiw2.close()
    

def calc_fpr_tpr(label, predict):
    fpr, tpr, _ = roc_curve(label, predict)
    return fpr, tpr

def get_precision_recall_f1_acc(label, predict):
    tp=0.0
    tn=0.0
    fp=0.0
    fn=0.0
    
#     np.savetxt('/media/user/Disk 02/wangzhiqin/TensorMulti/result/label.txt',label,fmt='%d')
#     np.savetxt('/media/user/Disk 02/wangzhiqin/TensorMulti/result/predict.txt',predict,fmt='%d')
    
#     print(label)
#     print(predict)
    
    predict = np.array(predict)   
    for i in range(predict.shape[0]):
        if label[i] == predict[i]:
            
            if label[i] == 1:
                tp = tp+1
               
            else:
                tn = tn+1
                
        else:
            if predict[i] == 1:
                fp = fp+1
                
            else:
                fn = fn+1
                 
    acc = (tp+tn)/(tp+tn+fn+fp) 
    if tp+fp == 0:
        precision = 0
    else:
        precision = tp/(tp+fp)
        
    if tp+fn == 0:
        recall = 0   #sensitivity
    else:
        recall = tp/(tp+fn)
    if tp == 0:
        f1 = 0
    else:
        f1 = 2*precision*recall/(precision+recall)
    return precision, recall, f1, acc


def matthews_correlation(label,predict):
    tp=0
    tn=0
    fp=0
    fn=0
    print('test number:',predict.shape[0])
    for i in range(predict.shape[0]):
        if label[i] == predict[i]:
            if label[i] == 1:
                tp = tp+1
            else:
                tn = tn+1
    else:
        if predict[i] == 1:
            fp = fp+1
        else:
            fn = fn+1
    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / (denominator + K.epsilon())
        
                
        
    

def load_mRNA():
#     X = h5py.File('mRNA_data_slct2000.mat')
    X = h5py.File('brc_mRNA_data_slct300.mat')  #tcga-350patients-brc
#     X = h5py.File('mRNA_data_slct122-30.mat')
  
    X = X['mRNA_data_slct300']
    X = np.transpose(X)  # [122,784]
    print('input data shape', X.shape)
#     Y = h5py.File('label_122-1.mat')
    Y = h5py.File('label350-251.mat')
#     Y = Y['label']
    Y = Y['label']
    Y = np.transpose(Y)  #[122,1]
    print('input label shape', Y.shape)
    #Y = tf.cast(Y, dtype = tf.float32)
   
    num_feature = 300
    num_patient = 350
    n_train = num_patient // 2  # 61
    n_val = num_patient // 10  # 12
    n_test = num_patient - n_train - n_val
    x_train = X[: n_train]
    y_train = Y[: num_patient-n_train]
    y_train = y_train[:,0]
    
## balance data
    index = np.where(y_train == 1)
   
    copy_data = x_train[index,:]
    print('copy_data:',copy_data.shape)
#     copy_data = np.reshape(copy_data, [20,num_feature])
    copy_data = np.reshape(copy_data, [136,num_feature])  #tcga_brc
    x_train = np.concatenate((x_train,copy_data),axis=0)
    copy_y = y_train[index]
#     copy_y = copy_y.reshape(20)
    copy_y = copy_y.reshape(136)  #tcga-bcr
    y_train = np.concatenate((y_train,copy_y),axis=0) #81
    
####  
    
    
    
    y_train = y_train.reshape((n_train+136)).astype(np.int32)
    
    x_train = np.reshape(x_train, [n_train+136,num_feature])
#     x_train = np.reshape(x_train, [n_train,num_feature,1])
#     x_train = x_train.reshape((n_train, 28, 28, 1)).astype(np.float32)
    print('input x_train type', type(x_train))

    x_val = X[n_train:n_train+n_val]
    y_val = Y[n_train:n_train+n_val]
    y_val = y_val[:, 0]
    y_val = y_val.reshape((n_val)).astype(np.int32)
#     x_val = x_val.reshape((n_val, 28, 28, 1)).astype(np.float32)
    x_val = np.reshape(x_val, [n_val,num_feature])
#     x_val = np.reshape(x_val, [n_val,num_feature,1])


    x_test = X[n_train+n_val:]
    y_test = Y[n_train+n_val:]
    y_test = y_test[:, 0]
    y_test = y_test.reshape((n_test)).astype(np.int32)
    x_test = np.reshape(x_test, [n_test, num_feature])
#     x_test = np.reshape(x_test, [n_test, num_feature, 1])
#     x_test = x_test.reshape((n_test, 28, 28, 1)).astype(np.float)
    print('x_val shape type', x_val.shape,type(x_val))
    print('y_train type',type(y_train))
    
    
    y_train_onehot = to_categorical(y_train)
    y_val_onehot = to_categorical(y_val)
#     print('y_val_onehot',y_val_onehot)
    y_test_onehot = to_categorical(y_test)

    y_test_onehot_feature = np.append(y_test_onehot, x_test[:,1])

    
    return x_train,x_val,x_test,y_train,y_val,y_test,y_train_onehot,y_val_onehot,y_test_onehot


def cv_load_mRNA():

    X = h5py.File('brc_mRNA_data_slct300.mat')  #tcga-350patients-brc  
    X = X['mRNA_data_slct300']
    X = np.transpose(X)  # [350,300]
    print('input data shape', X.shape)

    Y = h5py.File('label350-251.mat')
    Y = Y['label']
    Y = np.transpose(Y)  #[350,1]
    print('input label shape', Y.shape)   
    return X, Y
    
def binary_focal_loss(gamma=2, alpha=0.25):
    """
    Binary form of focal loss.
   
    
    focal_loss(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
        where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_true = tf.cast(y_true, tf.float32)
        alpha_t = y_true*alpha + (K.ones_like(y_true)-y_true)*(1-alpha)
    
        p_t = y_true*y_pred + (K.ones_like(y_true)-y_true)*(K.ones_like(y_true)-y_pred) + K.epsilon()
        focal_loss = - alpha_t * K.pow((K.ones_like(y_true)-p_t),gamma) * K.log(p_t)
        return K.mean(focal_loss)
    return binary_focal_loss_fixed

def plot_acc_loss(train_acc, val_acc, train_loss, val_loss,mode):
    # *****************
    # marker: '-o':圆形  '-s'：正方形  '-p'：五边形  '-v'：三角形
    # mode =1: 只画一条
    # mode = 0： 画出全部
    # *****************
    
    if mode == 0:        
        len_train = len(train_acc)
        len_val = len(val_acc)
        if len_train != len_val:
            print(len_train)
            print(val_acc)
            print('wrong input!')
            exit()
        for i in range(len_train):
            
            plt.clf()
            plt.plot(train_acc[i], '-o',label = 'Train_'+ str(i) + '_acc') # 圆形
            plt.plot(train_loss[i], '-s', label = 'Train_'+ str(i) + '_loss') # 正方形
        
            plt.plot(val_acc[i], '-p',label = 'val_'+ str(i) + '_acc')
            plt.plot(val_loss[i], '-v',label = 'val_'+ str(i) + '_loss')   
    
            plt.title('Model accuracy_loss')
            plt.ylabel('Accuracy/loss')
            plt.xlabel('Epoch')
            plt.legend(ncol = 5, loc='upper center')
            plt.savefig('/media/user/Disk 02/wangzhiqin/TensorMulti/result/inter_intra/' + 'cv_' + str(i) +'_acc_loss.png')  
    elif mode == 1:
        plt.plot(train_acc[2], label = 'Train'+ '_acc') 
        plt.plot(train_loss[2], label = 'Train' + '_loss') 
        plt.plot(val_acc[2], label = 'val'+  '_acc')
        plt.plot(val_loss[2], label = 'val'+  '_loss')   
        plt.title('Model accuracy_loss')
        plt.ylabel('Accuracy/loss')
        plt.xlabel('Epoch')
        plt.legend(loc='upper left')
        plt.savefig('/media/user/Disk 02/wangzhiqin/TensorMulti/result/acc_loss_1.png')   
        
def get_label():
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
    return label

def get_os_time():
    path = '/media/user/Disk 02/wangzhiqin/TensorMulti/data/cv_data/os_time.xls'
    data = xlrd.open_workbook(path)
    table = data.sheet_by_name('sheet1')
    colum = table.ncols
    row = table.nrows
    patientID = []
    os_time = []
    for i in range(1,row):
        os_time.append(table.cell(i,1).value)
        patientID.append(table.cell(i, 0).value)
    return os_time

def get_state():
    path = '/media/user/Disk 02/wangzhiqin/TensorMulti/data/cv_data/os_state.xls'
    data = xlrd.open_workbook(path)
    table = data.sheet_by_name('sheet1')
    colum = table.ncols
    row = table.nrows
    patientID = []
    os_state = []
    for i in range(1,row):
        os_state.append(table.cell(i,1).value)
        patientID.append(table.cell(i, 0).value)
    return os_state

def prepare_km(path,name_label,name_time,name_state,save_path,save_name):
    os_time = np.load(path+name_time)
    predict = np.load(path+name_label)
    state = np.load(path+name_state)
    
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet('sheet1')
    for i in range(len(os_time)):
        sheet.write(i,0,os_time[i])
    for i in range(len(os_time)):
        sheet.write(i,1,predict[i])
    for i in range(len(os_time)):
        sheet.write(i,2,state[i])
    workbook.save(save_path+save_name)  
    
def sample_mean(data):
    ##输入(x,y)  batch_size,64
    num = sum(X[:, None] for X in data)
    return num/64

def sample_variance(data):
    EX2 = sample_mean(data)**2
    E2X = sample_mean(x**2 for x in data)
    return E2X-EX2

 

    
    
# def plot_multi_roc():


# get fc1 weight
# def draw_weight():