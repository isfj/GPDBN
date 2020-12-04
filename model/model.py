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

def GPDFN():
    # **********
    #our proposed GPDBN framework
    # *********
    
    # **************
    #input：inputs=[input_gene, input_patho]
    # output：softmax score
    # input_gene, input_patho represent genomicd data and pathological images with the dimension of 32 respectively
    # **************
    input_gene = Input(shape = (32, ), name = 'gene_input')
    input_patho = Input(shape = (32, ), name = 'patho_input')
    
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
       
    x = Dense(500, activation='relu',name = 'FC_1')(inter_intra)  
      
    x = Dropout(0.1)(x)
    x = Dense(256, activation='relu', name = 'FC_5')(x)
    x = Dropout(0.1)(x)    
    x = Dense(128 , activation='relu', name = 'FC_3')(x)
    x = Dropout(0.1)(x)    
    x = Dense(32, activation='relu', name = 'FC_4')(x)
    x = Dropout(0.1)(x)    
    outputs = Dense(2, activation='softmax',name='output')(x)
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
        
def concat_model():
    inputs = Input(shape = (64,),name='input')
    x = Dense(500, activation='relu')(inputs)
    x = Dropout(0.3)(x) 
    x = Dense(256, activation='relu',kernel_regularizer=regularizers.l2(0.00001))(x)
    x = Dropout(0.3)(x)    
    x = Dense(128, activation='relu',kernel_regularizer=regularizers.l2(0.00001))(x)
    x = Dropout(0.3)(x)    
    x = Dense(32, activation='relu')(x)
    
    score = Dense(2, activation='softmax',name='output')(x)
    outputs = Concatenate(axis=1)([score,inputs])
    model = Model(inputs=inputs, outputs=outputs)
    return model


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

    
    
