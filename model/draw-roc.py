# -*- coding:utf-8 -*-
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve

from sklearn.model_selection import StratifiedKFold
import h5py
import numpy as np
from keras.utils.np_utils import to_categorical
from keras import backend as K
import tensorflow as tf
from compiler.ast import flatten
from utils import get_label,get_os_time, get_state,prepare_km
import time
from lifelines.utils import concordance_index
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def ROC(label,predict,path1,path2,name):
    fpr, tpr, threshold = roc_curve(label, predict)
    sp=1-fpr
    np.savetxt(path1+'sp.txt',sp,fmt='%.4f')
    np.savetxt(path1+'fpr.txt',fpr,fmt='%.4f')
    np.savetxt(path1+'tpr.txt',tpr,fmt='%.4f')
    np.savetxt(path1+'threshold.txt',threshold,fmt='%.4f')
    AUC = auc(fpr, tpr)
   
    plt.figure()
    plt.plot(fpr, tpr, color="red", label='ROC curve(auc = %0.2f)' % AUC)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title('TCGA-BRC-ROC')
    plt.legend(loc='lower right')
#     plt.savefig(path2+name)
   
    return AUC

def load_data(path,name1,name2):
    label = np.load(path+name1)
    predict = np.load(path+name2)
    return label,predict
    

mRNA_only_path = '/media/user/Disk 02/wangzhiqin/TensorMulti/result/mRNA-only/roc/'
patho_only_path = '/media/user/Disk 02/wangzhiqin/TensorMulti/result/patho-only/roc/'
intra_patho_path = '/media/user/Disk 02/wangzhiqin/TensorMulti/result/intra_patho/roc/'
intra_mRNA_path = '/media/user/Disk 02/wangzhiqin/TensorMulti/result/intra_mRNA/roc/'
cancatenate_path = '/media/user/Disk 02/wangzhiqin/TensorMulti/result/concatenate/roc/'
inter_mRNA_path = '/media/user/Disk 02/wangzhiqin/TensorMulti/result/inter_mRNA/roc/'
# inter_patho_path = '/media/user/Disk 02/wangzhiqin/TensorMulti/result/inter_patho/roc/'
GPDBFN_path = '/media/user/Disk 02/wangzhiqin/TensorMulti/result/inter_intra/roc/auc0.816/'


save_path = '/media/user/Disk 02/wangzhiqin/TensorMulti/result/roc/'
name = 'roc_final.png'












###ROC
cancatenate_ori_label,cancatenate_predict_score = load_data(cancatenate_path, 'cancatenate_ori_label.npy', 'cancatenate_predict_score.npy')

intra_patho_ori_label, intra_patho_predict_score = load_data(intra_patho_path, 'intra_patho_ori_label.npy', 'intra_patho_predict_score.npy')
intra_mRNA_ori_label, intra_mRNA_predict_score = load_data(intra_mRNA_path, 'intra_mRNA_ori_label.npy', 'intra_mRNA_predict_score.npy')

mRNA_only_ori_label, mRNA_only_predict_score = load_data(mRNA_only_path, 'mRNA_only_ori_label.npy','mRNA_only_predict_score.npy')
patho_only_ori_label, patho_only_predict_score = load_data(patho_only_path,'patho_only_ori_label.npy','patho_only_predict_score.npy')


inter_mRNA_ori_label, inter_mRNA_predict_score = load_data(inter_mRNA_path, 'inter_mRNA_ori_label.npy','inter_mRNA_predict_score.npy')

GPDBFN_ori_label, GPDBFN_predict_score = load_data(GPDBFN_path, 'inter_intra_ori_label.npy', 'inter_intra_predict_score.npy')




fpr_mRNA, tpr_mRNA, threshold_mRNA = roc_curve(mRNA_only_ori_label, mRNA_only_predict_score)
AUC_mRNA = auc(fpr_mRNA, tpr_mRNA)
fpr_patho, tpr_patho, threshold_patho = roc_curve(patho_only_ori_label, patho_only_predict_score)
AUC_patho = auc(fpr_patho, tpr_patho)

fpr_cancatenate, tpr_cancatenate, threshold_cancatenate = roc_curve(cancatenate_ori_label, cancatenate_predict_score)
AUC_cancatenate = auc(fpr_cancatenate, tpr_cancatenate)

fpr_GPDBFN, tpr_GPDBFN, threshold_GPDBFN = roc_curve(GPDBFN_ori_label, GPDBFN_predict_score)
AUC_GPDBFN = auc(fpr_GPDBFN, tpr_GPDBFN)


fpr_intra_patho, tpr_intra_patho, threshold_intra_patho = roc_curve(intra_patho_ori_label, intra_patho_predict_score)
AUC_intra_patho = auc(fpr_intra_patho, tpr_intra_patho)
fpr_intra_mRNA, tpr_intra_mRNA, threshold_intra_mRNA = roc_curve(intra_mRNA_ori_label, intra_mRNA_predict_score)
AUC_intra_mRNA = auc(fpr_intra_mRNA, tpr_intra_mRNA)

fpr_inter_mRNA, tpr_inter_mRNA, threshold_inter_mRNA = roc_curve(inter_mRNA_ori_label, inter_mRNA_predict_score)
AUC_inter_mRNA = auc(fpr_inter_mRNA, tpr_inter_mRNA)
               


plt.figure()
plt.plot(fpr_mRNA, tpr_mRNA, color = 'bisque',linestyle = '--', label='Baseline_G(auc = %0.3f)' % AUC_mRNA)
plt.plot(fpr_patho, tpr_patho, color = 'lightgreen',linestyle = '--', label='Baseline_P(auc = %0.3f)' % AUC_patho)

plt.plot(fpr_intra_mRNA, tpr_intra_mRNA,color = 'darkgoldenrod',linestyle = '-', label='Intra-BFEM_G(auc = %0.3f)' % AUC_intra_mRNA)
plt.plot(fpr_intra_patho, tpr_intra_patho, color = 'darkgreen',linestyle = '-',label='Intra-BFEM_P(auc = %0.3f)' % AUC_intra_patho)
plt.plot(fpr_cancatenate, tpr_cancatenate, color = 'cornflowerblue',linestyle = '--',label='Baseline_GP(auc = %0.3f)' % AUC_cancatenate)
plt.plot(fpr_inter_mRNA, tpr_inter_mRNA, color = 'darkviolet',linestyle = '-',label='Inter-BFEM*(auc = %0.3f)' % AUC_inter_mRNA)
plt.plot(fpr_GPDBFN, tpr_GPDBFN, lw =2,color = 'red',linestyle = '-',label='GPDBN(auc = %0.3f)' % AUC_GPDBFN)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig('/media/user/Disk 02/wangzhiqin/TensorMulti/result/roc/roc_oblation.png')


####PR-Curve
# precision_GPDBFN, recall_GPDBFN, thresholds_GPDBFN = precision_recall_curve(GPDBFN_ori_label, GPDBFN_predict_score)
# precision_cancatenate, recall_cancatenate, thresholds_cancatenate = precision_recall_curve(cancatenate_ori_label, cancatenate_predict_score)
# precision_mRNA_only, recall_mRNA_only, thresholds_mRNA_only = precision_recall_curve(mRNA_only_ori_label, mRNA_only_predict_score)
# precision_patho_only, recall_patho_only, thresholds_patho_only = precision_recall_curve(patho_only_ori_label, patho_only_predict_score)
# precision_intra_mRNA, recall_intra_mRNA, thresholds_intra_mRNA = precision_recall_curve(intra_mRNA_ori_label, intra_mRNA_predict_score)
# precision_intra_patho, recall_intra_patho, thresholds_intra_patho = precision_recall_curve(intra_patho_ori_label, intra_patho_predict_score)
# precision_inter_mRNA, recall_inter_mRNA, thresholds_inter_mRNA = precision_recall_curve(inter_mRNA_ori_label, inter_mRNA_predict_score)


# plt.figure(1) # 创建图表1
# plt.title('Precision/Recall Curve')# give plot a title
# plt.xlabel('Recall')# make axis labels
# plt.ylabel('Precision')

# # plt.plot(precision_GPDBFN, recall_GPDBFN,label='GPDBFN')
# # plt.plot(precision_cancatenate, recall_cancatenate,label='cancatenate') 
# plt.plot(precision_mRNA_only, recall_mRNA_only,label='mRNA_only')
# plt.plot(precision_patho_only, recall_patho_only,label='patho_only')
# plt.plot(precision_intra_mRNA, recall_intra_mRNA,label='intra_mRNA')
# plt.plot(precision_intra_patho, recall_intra_patho,label='intra_patho')
# plt.plot(precision_inter_mRNA, recall_inter_mRNA,label='inter_mRNA_patho')
# plt.legend(loc='lower right')
# plt.savefig(save_path + 'p-r.png')
# # plt.savefig('/media/user/Disk 02/wangzhiqin/TensorMulti/result/inter_intra/p-r.png')


###KM
# km_path = '/media/user/Disk 02/wangzhiqin/TensorMulti/comparison_result/deepcorrsurv/km/'
# name_label = 'deepcorrsurv_predict_label.npy'
# name_time = 'deepcorrsurv_os_time.npy'
# name_state = 'deepcorrsurv_state.npy'
# save_name = 'deepcorrsurv_km.xls'
# prepare_km(km_path,name_label,name_time,name_state,km_path,save_name)


# km_path = '/media/user/Disk 02/wangzhiqin/TensorMulti/result/inter_intra/km/2020_11_13/'
# name_label = 'inter_intra_predict_label.npy'
# name_time = 'inter_intra_os_time.npy'
# name_state = 'inter_intra_state.npy'
# save_name = 'GPDBFN_km.xls'
# prepare_km(km_path,name_label,name_time,name_state,km_path,save_name)



