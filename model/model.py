# -*- coding:utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input, Dropout,Reshape,merge,Add,Concatenate
from keras.models import Model
import tensorflow as tf
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from sklearn.model_selection import StratifiedKFold, train_test_split
import numpy as np
from utils import *
import xlrd
from keras import backend as k
from keras.layers.core import Lambda

def load_data():
	train_gene = np.load('/media/user/Disk 02/wangzhiqin/TensorMulti/data/train/train_gene.npy')
	val_gene = np.load('/media/user/Disk 02/wangzhiqin/TensorMulti/data/val/val_gene.npy')
	test_gene = np.load('/media/user/Disk 02/wangzhiqin/TensorMulti/data/test/test_gene.npy')
	
	train_patho = np.load('/media/user/Disk 02/wangzhiqin/TensorMulti/data/train/train_patho.npy')
	val_patho = np.load('/media/user/Disk 02/wangzhiqin/TensorMulti/data/val/val_patho.npy')
	test_patho = np.load('/media/user/Disk 02/wangzhiqin/TensorMulti/data/test/test_patho.npy')
	
	train_label = np.load('/media/user/Disk 02/wangzhiqin/TensorMulti/data/train/trainLabel.npy')
	val_label = np.load('/media/user/Disk 02/wangzhiqin/TensorMulti/data/val/valLabel.npy')
	test_label = np.load('/media/user/Disk 02/wangzhiqin/TensorMulti/data/test/testLabel.npy')
	return train_gene, val_gene, test_gene,train_patho,val_patho,test_patho,train_label,val_label,test_label

def load_gege():
	dataPath = '/media/user/Disk 02/wangzhiqin/TensorMulti/data/origin/mRNA/mRNA_data_slct32.xlsx'
	data = xlrd.open_workbook(dataPath)
	table = data.sheet_by_name('Sheet1')
	colum = table.ncols
	row = table.nrows
	data_mRNA = []
	for i in range(row):
		data_mRNA.append(table.row_values(i))
	return data_mRNA

def load_patho():
	dataPath = '/media/user/Disk 02/wangzhiqin/TensorMulti/data/origin/patho/patho_data_slct32.xlsx'
	data = xlrd.open_workbook(dataPath)
	table = data.sheet_by_name('Sheet1')
	colum = table.ncols
	row = table.nrows
	data_patho = []
	for i in range(row):
		data_patho.append(table.row_values(i))
	return data_patho

def SALMON():
    inputs = Input(shape = (300,))
    x = Dense(8, activation='sigmoid')(inputs)
#     x = Dense(4, activation='sigmoid')(x)
    outputs = Dense(2, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def MDNNMD():
    inputs = Input(shape = (32,),name='input')
  
    x = Dense(1000, activation='tanh')(inputs)
    x = Dropout(0.5)(x)
    x = Dense(500, activation='tanh')(x)
    x = Dropout(0.5)(x)
    x = Dense(500, activation='tanh')(x)
    x = Dropout(0.5)(x)
    x = Dense(100, activation='tanh')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(2, activation='sigmoid',name='output')(x)
  
    model = Model(inputs=inputs, outputs=outputs)    
    return model
    
def DeepSurv():    
    inputs = Input(shape = (32,),name='input')
    x = Dense(600, activation='relu')(inputs)
    x = Dropout(0.16)(x)
    x = Dense(500, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.16)(x)
    x = Dense(32, activation='relu')(x)
    
    '''
    x = Dense(28, activation='relu',kernel_regularizer=regularizers.l2(0.0001),name='hidden1')(inputs)
    x = Dense(10, activation='relu',kernel_regularizer=regularizers.l2(0.0001),name='hidden2')(x)
    x = BatchNormalization(axis=1)(x)
    x = Dropout(0.5)(x)
   '''
    outputs = Dense(2, activation='sigmoid',name='output')(x)
    model = Model(inputs=inputs, outputs=outputs)    
    
    return model

def intra_fusion():
    # *************
    # 考虑单模态+模态内
    # *************
    input_gene = Input(shape = (32, ), name = 'gene_input')
    gene = Reshape((1, 32))(input_gene)
    gene_gene = merge([gene, gene], mode='dot', dot_axes=1, name='gene-gene')
    intra = Flatten()(gene_gene)
    intra = Dense(32, activation='relu')(intra)
    gene_intra = merge([input_gene, intra], mode='concat', dot_axes=1, name='gene_intra')
    
    x = Dense(600,activation='relu',name = 'FC_1')(gene_intra)
    x = Dropout(0.2)(x)
    x = Dense(500, activation='relu',name = 'FC_2')(x)
    x = Dropout(0.2)(x)
    x = Dense(256, activation='relu',name = 'FC_3')(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu',name = 'FC_5')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu',name = 'FC_4')(x)
    outputs = Dense(2, activation='sigmoid',name='output')(x)
    model = Model(inputs=input_gene, outputs=outputs)
    return model

def inter_model_action():
    input_gene = Input(shape = (32, ), name = 'gene_input')
    input_patho = Input(shape = (32, ), name = 'patho_input')
    
    biased_gene = Reshape((1, 32))(input_gene)
    biased_patho = Reshape((1, 32 ))(input_patho)
        
    inter = merge([biased_gene, biased_patho], mode='dot', dot_axes=1, name='gene-patho')
    inter = Flatten()(inter)
#     inter = Dense(32, activation='relu')(inter)    

    x = Dense(500, activation='relu',name = 'FC_1')(inter)  
#     x = BatchNormalization(axis=1)(x)
    x = Dropout(0.2)(x)
#     x = Dense(64, activation='relu', name = 'FC_1')(inter_intra)
    x = Dense(256, activation='relu', name = 'FC_2')(x)
    x = Dense(128 , activation='relu', name = 'FC_3')(x)
    x = Dense(32, activation='relu', name = 'FC_4')(x)
#     x = Dense(32, activation='relu', name = 'FC_5')(x)
    outputs = Dense(2, activation='sigmoid',name='output')(x)
    model = Model(inputs=[input_gene, input_patho], outputs=outputs)      
    
    return model
    
    
    

def GPDFN():
    # **********
    #最终的模型
    # *********
    
    # **************
    # 输入：inputs=[input_gene, input_patho]
    # 输出：sigmoid score
    # input_gene, input_patho分别为输入的基因表达和病理特征，32维
    # **************
    input_gene = Input(shape = (32, ), name = 'gene_input')
    input_patho = Input(shape = (32, ), name = 'patho_input')
#     input_data =  merge([input_gene, input_patho], mode='concat', dot_axes=1, name='input')
    
    ####sim loss
#     mean = Lambda(lambda x: sample_mean(x))(input_data)
#     mean = sample_mean((input_gene, input_patho))
#     var = tf.reduce_mean(sample_variance((input_gene, input_patho)))
#     var2 = sample_mean(((input_gene - tf.reduce_mean(mean))**2, (input_patho - tf.reduce_mean(mean))**2))
#     ratios = var/tf.reduce_mean(var2,1)
#     ratio = tf.reduce_mean(tf.clip_by_value(ratios, 0.02,1.0))
   
    #ratio = ratio.clamp(min = 0.02, max = 1.0).mean()
    ####
    
    biased_gene = Reshape((1, 32))(input_gene)
    biased_patho = Reshape((1, 32 ))(input_patho)
        
    inter = merge([biased_gene, biased_patho], mode='dot', dot_axes=1, name='gene-patho')
    inter = Flatten()(inter)
    inter = Dense(20, activation='relu',kernel_regularizer=regularizers.l2(0.00001),name='gene_patho_layer')(inter)    
    
    gene_gene = merge([biased_gene, biased_gene], mode='dot', dot_axes=1, name='gene-gene')
    gene_gene = Flatten()(gene_gene)
    intra_gene = Dense(20, activation='relu',kernel_regularizer=regularizers.l2(0.00001),name='gene_gene_layer')(gene_gene)
  
    
    
    patho_patho = merge([biased_patho, biased_patho], mode='dot', dot_axes=1, name='patho-patho')
    patho_patho = Flatten()(patho_patho)
    intra_patho = Dense(20, activation='relu',kernel_regularizer=regularizers.l2(0.00001),name='patho_patho_layer')(patho_patho)  

    
    inter_intra = merge([input_gene, input_patho, inter, intra_gene, intra_patho], mode='concat', dot_axes=1, name='inter-intra')
    

#     x = Dense(1000, activation='relu',name = 'FC_1')(inter_intra)       
#     x = Dropout(0.3)(x)
#     x = Dense(500, activation='relu', name = 'FC_5')(x)
#     x = Dropout(0.3)(x)    
#     x = Dense(500 , activation='relu', name = 'FC_3')(x)
#     x = Dropout(0.3)(x)    
#     x = Dense(100, activation='relu', name = 'FC_4')(x)
#     x = Dropout(0.3)(x)     
    
    
    
    
    
    x = Dense(500, activation='relu',name = 'FC_1')(inter_intra)  
      
    x = Dropout(0.1)(x)
    x = Dense(256, activation='relu', name = 'FC_5')(x)
    x = Dropout(0.1)(x)    
    x = Dense(128 , activation='relu', name = 'FC_3')(x)
    x = Dropout(0.1)(x)    
    x = Dense(32, activation='relu', name = 'FC_4')(x)
    x = Dropout(0.1)(x)    
#     score = Dense(2, activation='sigmoid',name='output')(x)
#     cca_input = [input_gene, input_patho]
#     outputs = Concatenate(axis=1)([score,cca_input]) 
    outputs = Dense(2, activation='sigmoid',name='output')(x)
    model = Model(inputs=[input_gene, input_patho], outputs=outputs)      
    
    return model


def tensorfusion():
    
    # *********
    # 加偏置
    # *********    
    
    input_gene = Input(shape = (32, ), name = 'gene_input')
    input_patho = Input(shape = (32, ), name = 'patho_input')
    
    # ********
    # 加上偏置 33*33
    # ********
    
    bias_model = Input(shape=(1,), name='bias')
    biased_gene = merge([bias_model, input_gene], mode='concat', name = 'bias_gene')
    biased_gene = Reshape((1, 32 + 1))(biased_gene)
    biased_patho = merge([bias_model, input_patho], mode='concat', name = 'bias_patho')
    biased_patho = Reshape((1, 32 + 1))(biased_patho)
    dot_layer = merge([biased_gene, biased_patho], mode='dot', dot_axes=1, name='dot_layer')

    
    x = Flatten()(dot_layer) 
    
    x = Dense(600, activation='relu', name = 'FC_1')(x)                  
    x = Dense(500, activation='relu', name = 'FC_2')(x)
    x = Dense(256, activation='relu', name = 'FC_3')(x)
    x = Dense(128, activation='relu', name = 'FC_4')(x)
    x = Dense(32, activation='relu', name = 'FC_5')(x)
    outputs = Dense(2, activation='sigmoid',name='output')(x)
    model = Model(inputs=[input_gene, input_patho, bias_model], outputs=outputs)    
    return model


def inter_tensor_gene():
    # ********
    # gene+模态间交互
    # *********
    input_gene = Input(shape = (32, ), name = 'gene_input')
    input_patho = Input(shape = (32, ), name = 'patho_input')
    
    biased_gene = Reshape((1, 32))(input_gene)
    biased_patho = Reshape((1, 32 ))(input_patho)
    dot_layer = merge([biased_gene, biased_patho], mode='dot', dot_axes=1, name='dot_layer')
    
    x = Flatten()(dot_layer) 
    
    x = Dense(32, activation='relu', name = 'FC_1')(x)
    inter_intra = merge([x, input_gene], mode='concat', dot_axes=1, name='inter-gene')
    
    
    inter_intra = Dense(500, activation='relu', name = 'FC_2')(inter_intra)
    inter_intra = Dropout(0.1)(inter_intra)
    inter_intra = Dense(256, activation='relu', name = 'FC_3')(inter_intra)
    inter_intra = Dropout(0.1)(inter_intra)
    inter_intra = Dense(128, activation='relu', name = 'FC_4')(inter_intra)
    inter_intra = Dense(32, activation='relu', name = 'FC_5')(inter_intra)
    outputs = Dense(2, activation='sigmoid',name='output')(inter_intra)
    model = Model(inputs=[input_gene, input_patho], outputs=outputs)
  
    return model    

def inter_tensor_patho():
    # ********
    # patho+gene+模态间交互
    # *********
    input_gene = Input(shape = (32, ), name = 'gene_input')
    input_patho = Input(shape = (32, ), name = 'patho_input')
    
    biased_gene = Reshape((1, 32))(input_gene)
    biased_patho = Reshape((1, 32 ))(input_patho)
    dot_layer = merge([biased_gene, biased_patho], mode='dot', dot_axes=1, name='dot_layer')
    
    x = Flatten()(dot_layer) 
    
    x = Dense(32, activation='relu', name = 'FC_1')(x)
    inter_intra = merge([x, input_patho,input_gene], mode='concat', dot_axes=1, name='inter-patho')
    
    
    inter_intra = Dense(500, activation='relu', name = 'FC_2')(inter_intra)
#     x = BatchNormalization(axis=1)(x)
    x = Dropout(0.2)(x)   
    inter_intra = Dense(256, activation='relu', name = 'FC_6')(inter_intra)
    inter_intra = Dense(256, activation='relu', name = 'FC_3')(inter_intra)
    inter_intra = Dense(128, activation='relu', name = 'FC_4')(inter_intra)
    inter_intra = Dense(32, activation='relu', name = 'FC_5')(inter_intra)
    outputs = Dense(2, activation='sigmoid',name='output')(inter_intra)
    
    model = Model(inputs=[input_gene, input_patho], outputs=outputs)
  
    return model  

def get_cv_data():
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
    skf = StratifiedKFold(n_splits = 5)
    i = 1
    train_index = []
    test_index = []
    for train_indx, test_indx in skf.split(data_mRNA, label): 
        print(i,'fold #####')
        train_index.append(train_indx)
        test_index.append(test_indx)
        print(test_indx)
        i += 1
    print('\n')
    print(test_index[0])
    np.save('/media/user/Disk 02/wangzhiqin/TensorMulti/data/cv_data/train_index.npy', train_index)
    np.save('/media/user/Disk 02/wangzhiqin/TensorMulti/data/cv_data/test_index.npy', test_index)
        
def cnn_model():
    inputs = Input(shape = (1024, 1024, 3))
    x = Conv2D(32,(7,7),strides=(3,3),padding='same',activation='relu',name='conv-layer1')(inputs)
    x = MaxPooling2D(pool_size=(2,2), strides = None, padding = 'valid',name='maxpooling1')(x)
    x = Conv2D(32,(5,5),strides=(2,2),padding='same',activation='relu',name='conv-layer2')(x)
    x = Conv2D(32,(3,3),strides=(2,2),padding='same',activation='relu',name='conv-layer3')(x)
    x = MaxPooling2D(pool_size=(2,2), strides = None, padding = 'valid',name='maxpooling2')(x)
    
    flat = Flatten()(x)
    drop_1 = Dropout(0.5)(flat)
    outputs = Dense(32, activation='sigmoid',name='output')(drop_1)
    model = Model(inputs=inputs, outputs=outputs)
    return model


def fcn_model():
    inputs = Input(shape = (32,),name='input')
    x = Dense(600, activation='relu')(inputs)
    x = Dense(500, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    
    '''
    x = Dense(28, activation='relu',kernel_regularizer=regularizers.l2(0.0001),name='hidden1')(inputs)
    x = Dense(10, activation='relu',kernel_regularizer=regularizers.l2(0.0001),name='hidden2')(x)
    x = BatchNormalization(axis=1)(x)
    x = Dropout(0.5)(x)
   '''
    outputs = Dense(2, activation='sigmoid',name='output')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

def concat_model():
    inputs = Input(shape = (64,),name='input')
    x = Dense(500, activation='relu')(inputs)
    x = Dropout(0.3)(x) 
    x = Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.00001))(x)
    x = Dropout(0.3)(x)    
    x = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.00001))(x)
    x = Dropout(0.3)(x)    
    x = Dense(32, activation='relu')(x)
    
    '''
    x = Dense(28, activation='relu',kernel_regularizer=regularizers.l2(0.0001),name='hidden1')(inputs)
    x = Dense(10, activation='relu',kernel_regularizer=regularizers.l2(0.0001),name='hidden2')(x)
    x = BatchNormalization(axis=1)(x)
    x = Dropout(0.5)(x)
   '''
    score = Dense(2, activation='sigmoid',name='output')(x)
    outputs = Concatenate(axis=1)([score,inputs])
#     outputs = Dense(2, activation='sigmoid',name='output')(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model



def deepcorrsurv():
    input_gene = Input(shape = (32, ), name = 'gene_input')
    input_patho = Input(shape = (32, ), name = 'patho_input') 
    gene = Dense(32, activation='relu',name = 'FC_gene')(input_gene)
    patho = Dense(32, activation='relu',name = 'FC_patho')(input_patho)
    input_cca = merge([gene, patho], mode='concat', dot_axes=1, name='input_cca')
    cca = Dense(32, activation='relu',name = 'FC_1')(input_cca)
    cca = Dense(16, activation='relu',name = 'FC_2')(cca)
    output = Dense(2, activation='sigmoid', name = 'output')(cca)
    model = Model(inputs=[input_gene, input_patho], outputs=[input_cca, output])
    return model
    
def deepcca():
    input_gene = Input(shape = (32, ), name = 'gene_input')
    input_patho = Input(shape = (32, ), name = 'patho_input') 
    gene = Dense(32, activation='relu',name = 'FC_gene')(input_gene)
    patho = Dense(32, activation='relu',name = 'FC_patho')(input_patho)
    input_cca = merge([gene, patho], mode='concat', dot_axes=1, name='input_cca')
    cca = Dense(32, activation='relu',name = 'FC_1')(input_cca)
    cca = Dense(16, activation='relu',name = 'FC_2')(cca)
    output = Dense(2, activation='sigmoid', name = 'output')(cca)
    model = Model(inputs=[input_gene, input_patho], outputs=output)
    return model
#############loss

# def corr_loss(y_true, cca_input):
#     x = cca_input['input_cca']
#     x = cca_input[:,2:]
#     mean_x = tf.reduce_mean(x, axis=0)
#     x = x - mean_x
#     instance = tf.shape(x)[0]
#     dim = tf.shape(x)[1]
#     vec_size = dim // 2
#     a = tf.cast(x[:, 0:vec_size], tf.float32)
#     b = tf.cast(x[:, vec_size:dim], tf.float32)
#     c = tf.reduce_sum(tf.multiply(a, b))
#     d = tf.sqrt(tf.reduce_sum(tf.multiply(a, a)) + tf.reduce_sum(tf.multiply(b, b)))+1e-12
    
#     return 0.3*(c / d) + K.binary_crossentropy(y_true, cca_input[:, 0:2])

def corr_loss(y_true, cca_input):
    x = cca_input
    mean_x = tf.reduce_mean(x, axis=0)
    x = x - mean_x
    instance = tf.shape(x)[0]
    dim = tf.shape(x)[1]
    vec_size = dim // 2
    a = tf.cast(x[:, 0:vec_size], tf.float32)
    b = tf.cast(x[:, vec_size:dim], tf.float32)
    c = tf.reduce_sum(tf.multiply(a, b))
    d = tf.sqrt(tf.reduce_sum(tf.multiply(a, a)) + tf.reduce_sum(tf.multiply(b, b)))+1e-12
    
    return -(c / d)

    
    