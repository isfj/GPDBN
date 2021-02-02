# -*- coding:utf-8 -*-
import os
import logging
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt
import torch

# Env
from data_loaders import *
from options import parse_args
from train_test import train, test
from utils import CIndex_lifeline
from lifelines.utils import concordance_index


def clc_cindex(label,predict,censor):
    cindex = concordance_index(label,predict,censor)
#     print('cindex=',cindex)
    return cindex


### 1. Initializes parser and device
opt = parse_args()
device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
print("Using device:", device)
if not os.path.exists(os.path.join(opt.model_save, opt.exp_name, opt.model_name)):
        os.makedirs(os.path.join(opt.model_save, opt.exp_name, opt.model_name))

### 2. Initializes Data
data_cv_path = '%s%s' % (opt.dataroot,opt.datatype)
print("Loading %s" % data_cv_path)
data_cv = pickle.load(open(data_cv_path, 'rb'))
data_cv_splits = data_cv['cv_splits']
results = []
pre_score = []
censored = []
ori_sur = []
### 3. Sets-Up Main Loop
for k, data in data_cv_splits.items():
    print("*******************************************")
    print("************** SPLIT (%d/%d) **************" % (k, len(data_cv_splits.items())))
    print("*******************************************")

    ### ### ### ### ### ### ### ### ###创建文件夹存储结果### ### ### ### ### ### ### ### ### ###
    if not os.path.exists(os.path.join(opt.results,opt.exp_name, opt.model_name,'%d_fold'%(k))): os.makedirs(
        os.path.join(opt.results,opt.exp_name, opt.model_name,'%d_fold'%(k)))

    ### 3.1 Trains Model
    model, optimizer, metric_logger = train(opt, data, device, k)
    epochs_list = range(opt.epoch_count, opt.niter+opt.niter_decay+1)

    ### 3.2 Evalutes Train + Test Error, and Saves Model
    loss_train, cindex_train, pvalue_train, surv_acc_train, pred_train = test(opt, model, data, 'train', device)
    loss_test, cindex_test, pvalue_test, surv_acc_test, pred_test = test(opt, model, data, 'test', device)  #pred_test = [risk_pred_all, survtime_all, censor_all]

    print("[Final] Apply model to training set: C-Index: %.10f, P-Value: %.10e" % (cindex_train, pvalue_train))
    logging.info("[Final] Apply model to training set: C-Index: %.10f, P-Value: %.10e" % (cindex_train, pvalue_train))
    print("[Final] Apply model to testing set: C-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
    logging.info("[Final] Apply model to testing set: cC-Index: %.10f, P-Value: %.10e" % (cindex_test, pvalue_test))
    # results.append(np.max(metric_logger['test']['cindex']))
    results.append(cindex_test)
    pre_score.extend(-pred_test[0])
    ori_sur.extend(pred_test[1])
    censored.extend(pred_test[2])



    print()

#     pickle.dump(pred_train, open(os.path.join(opt.results,opt.exp_name, opt.model_name,'%d_fold'%(k), '%s_%dpred_train.pkl' % (opt.model_name, k)), 'wb'))
#     pickle.dump(pred_test, open(os.path.join(opt.results,opt.exp_name, opt.model_name,'%d_fold'%(k), '%s_%dpred_test.pkl' % (opt.model_name, k)), 'wb'))


print('Split Results:', results)
print("Average:", np.array(results).mean())

print('concatenate cindex:', clc_cindex(ori_sur,pre_score,censored))
    # save for KM
km_path = '/media/user/Disk 02/wangzhiqin/TensorMulti/GPDBN_pytorch/km/'
np.save(km_path+'pre_score.npy',pre_score)
np.save(km_path+'ori_sur.npy', ori_sur)  
np.save(km_path+'censored.npy', censored)
# pickle.dump(results, open(os.path.join(opt.results, opt.exp_name, opt.model_name, '%s_results.pkl' % opt.model_name), 'wb'))