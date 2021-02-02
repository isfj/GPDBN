from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init, Parameter
import torch.optim.lr_scheduler as lr_scheduler
import math

def define_optimizer(opt, model):
    optimizer = None
    if opt.optimizer_type == 'adabound':
        optimizer = adabound.AdaBound(model.parameters(), lr=opt.lr, final_lr=0.1)
    elif opt.optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), weight_decay=opt.weight_decay)
    elif opt.optimizer_type == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay, initial_accumulator_value=0.1)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % opt.optimizer)
    return optimizer


def define_scheduler(opt, optimizer):
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, 0.1, last_epoch=-1)
    elif opt.lr_policy == 'step':
       scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
       scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
       scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
       return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler



class GPDBN(nn.Module):
    def __init__(self, in_size, output_dim, hidden=20, dropout=0.1):
        super(GPDBN, self).__init__()
        skip_dim =in_size*2

        self.gene_gene = nn.Sequential(nn.Linear(in_size*in_size, hidden), nn.ReLU())
        self.path_path = nn.Sequential(nn.Linear(in_size*in_size, hidden), nn.ReLU())
        self.gene_path = nn.Sequential(nn.Linear(in_size*in_size, hidden), nn.ReLU())

        encoder1 = nn.Sequential(nn.Linear((skip_dim+hidden*3), 500), nn.ReLU(), nn.Dropout(p=dropout))
        encoder2 = nn.Sequential(nn.Linear(500,256), nn.ReLU(), nn.Dropout(p=dropout))
        encoder3 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(p=dropout))
        encoder4 = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Dropout(p=dropout))
        self.encoder = nn.Sequential(encoder1,encoder2,encoder3,encoder4)
        self.classifier = nn.Sequential(nn.Linear(in_size,output_dim), nn.Sigmoid())

        self.output_range = Parameter(torch.FloatTensor([6]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-3]), requires_grad=False)


    def forward(self, x_gene,x_path):
        o1 = x_gene.squeeze(1)
        o2 = x_path.squeeze(1)
        o11 = torch.bmm(o1.unsqueeze(2), o1.unsqueeze(1)).flatten(start_dim=1)
        o22 = torch.bmm(o2.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)
        inter = self.gene_path(o12)
        intra_gene = self.gene_gene(o11)
        intra_path = self.path_path(o22)
        fusion = torch.cat((o1,o2,inter,intra_gene,intra_path),1)
        code = self.encoder(fusion)
        out =  self.classifier(code)
        out = out * self.output_range + self.output_shift
        return out, code


