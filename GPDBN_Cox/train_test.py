import random
from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from networks import define_scheduler, define_optimizer, GPDBN
from torch.utils.data import DataLoader
from data_loaders import graph_fusion_DatasetLoader
from utils import CoxLoss1, CoxLoss2, regularize_weights, CIndex_lifeline, cox_log_rank, accuracy_cox,count_parameters,CIndex
import torch.optim as optim
import pickle
import os
import gc


def train(opt,data,device,k):
    cudnn.deterministic = True
    torch.cuda.manual_seed_all(667)
    torch.manual_seed(667)
    random.seed(667)
    model = GPDBN(opt.input_size, opt.label_dim).to(device)
    optimizer = define_optimizer(opt, model)
    scheduler = define_scheduler(opt, optimizer)
    print(model)
    print("Number of Trainable Parameters: %d" % count_parameters(model))
    custom_data_loader = graph_fusion_DatasetLoader(data, split='train')
    train_loader = DataLoader(dataset = custom_data_loader, batch_size = len(custom_data_loader), num_workers = 4, shuffle = False)
    metric_logger = {'train':{'loss':[], 'pvalue':[], 'cindex':[], 'surv_acc':[]},
                      'test':{'loss':[], 'pvalue':[], 'cindex':[], 'surv_acc':[]}}
    c_index_best = 0

    for epoch in tqdm(range(opt.epoch_count, opt.niter+opt.niter_decay+1)):
        model.train()
        risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])
        loss_epoch = 0
        gc.collect()
        for batch_idx, (x_gene, x_path, x_cna,censor, survtime) in enumerate(train_loader):
            censor = censor.to(device)
            pred, _ = model(x_gene.to(device),x_path.to(device))
            loss_cox =CoxLoss2(survtime, censor, pred, device)
            loss_reg = regularize_weights(model=model)
            loss = loss_cox + opt.lambda_reg *  loss_reg
            loss_epoch += loss_cox.data.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()

            risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))
            censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))
            survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))
        scheduler.step()
        if opt.measure or epoch == (opt.niter+opt.niter_decay - 1):
            loss_epoch /= len(train_loader.dataset)
            cindex_epoch = CIndex_lifeline(risk_pred_all, censor_all, survtime_all)
            pvalue_epoch = cox_log_rank(risk_pred_all, censor_all, survtime_all)
            surv_acc_epoch = accuracy_cox(risk_pred_all, censor_all)
            loss_test, cindex_test, pvalue_test, surv_acc_test,pred_test = test(opt, model, data, 'test', device)

            metric_logger['train']['loss'].append(loss_epoch)
            metric_logger['train']['cindex'].append(cindex_epoch)
            metric_logger['train']['pvalue'].append(pvalue_epoch)
            metric_logger['train']['surv_acc'].append(surv_acc_epoch)

            metric_logger['test']['loss'].append(loss_test)
            metric_logger['test']['cindex'].append(cindex_test)
            metric_logger['test']['pvalue'].append(pvalue_test)
            metric_logger['test']['surv_acc'].append(surv_acc_test)

            pickle.dump(pred_test, open(os.path.join(opt.results,opt.exp_name,opt.model_name,'%d_fold'%(k), '%s_%d_pred_test.pkl' % (opt.model_name,epoch)), 'wb'))
            if cindex_test > c_index_best:
                c_index_best = cindex_test
            if opt.verbose > 0:
                print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}'.format('Train', loss_epoch, 'C-Index', cindex_epoch))
                print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}\n'.format('Test', loss_test, 'C-Index', cindex_test))

    return model, optimizer, metric_logger

def test(opt,model, data, split, device):
    model.eval()
    custom_data_loader = graph_fusion_DatasetLoader(data, split)
    test_loader = DataLoader(dataset = custom_data_loader,  batch_size = len(custom_data_loader), num_workers = 4, shuffle = False)
    risk_pred_all, censor_all, survtime_all = np.array([]), np.array([]), np.array([])
    loss_test = 0

    for batch_idx, (x_gene,x_path,x_cna, censor, survtime) in enumerate(test_loader):
        censor = censor.to(device)
        pred, _ = model(x_gene.to(device),x_path.to(device))
        loss_cox = CoxLoss2(survtime, censor, pred, device)
        loss_test += loss_cox.data.item()

        risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))
        censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))
        survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))

    ###################################################
    # ==== Measuring Test Loss, C-Index, P-Value ==== #
    ###################################################
    loss_test /= len(test_loader.dataset)
    cindex_test = CIndex_lifeline(risk_pred_all, censor_all, survtime_all)
    pvalue_test = cox_log_rank(risk_pred_all, censor_all, survtime_all)
    surv_acc_test = accuracy_cox(risk_pred_all, censor_all)
    pred_test = [risk_pred_all, survtime_all, censor_all]
    return loss_test, cindex_test, pvalue_test, surv_acc_test, pred_test