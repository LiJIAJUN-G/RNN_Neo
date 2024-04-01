# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 11:37:17 2023

@author: lijiajun
"""
import os
import torch
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import DataLoader
from torch import nn
from sklearn.metrics import roc_curve, auc, precision_recall_curve

log_file = open("RNN_train.log", "w+")

def makedata(type_, cols, batch_size, shuffle_):
    dat = pd.read_csv('./data/{}.csv'.format(type_))
    dat.replace({"response_type":{'negative':0,'CD8':1}},inplace=True)
    dat = dat[cols]
    dim = dat.shape[1] -3
    data = dat.iloc[:,2:-1].to_numpy().astype(np.float32).reshape((dat.shape[0],dim,1))
    labels = dat.iloc[:,-1].to_numpy().astype(np.float32).reshape((dat.shape[0],1))
    dataset = list(zip(data, labels))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle = shuffle_)
    return dat,data_loader



class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out


f_mean = lambda l: sum(l)/len(l)



def performances(y_true, y_prob, print_):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    if print_:
        log_file.write('auc={:.4f}|aupr={:.4f}'.format(roc_auc, pr_auc) + "\n")
        log_file.flush()
        print('auc={:.4f}|aupr={:.4f}'.format(roc_auc, pr_auc))
    return (roc_auc,pr_auc)

def evaluate_epoch(model, data_loader, loss):
    model.eval() 
    loss_val_list = []
    y_true_val_list, y_prob_val_list = [], []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            y_prob_val = model(data)
            y_true_val = target.cpu().numpy()
            val_loss = loss(y_prob_val, target)
            
            y_true_val_list.extend(y_true_val)
            y_prob_val_list.extend(y_prob_val.cpu().detach().numpy())
            loss_val_list.append(val_loss)
        ys_val = (y_true_val_list, y_prob_val_list)
        metrics_val = performances(y_true_val_list, y_prob_val_list, print_ = True)
    return ys_val, f_mean(loss_val_list).cpu().detach().numpy(), metrics_val



def train_epoch(model, data_loader, loss, updater):
    model.train()
    y_true_train_list, y_prob_train_list = [], []
    loss_train_list = []
    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        
        y_prob_train = model(data)
        
        train_loss = loss(y_prob_train, target)
        updater.zero_grad()
        train_loss.backward()
        updater.step()
        
        y_true_train = target.cpu().numpy()
        y_true_train_list.extend(y_true_train)
        y_prob_train_list.extend(y_prob_train.cpu().detach().numpy())
        loss_train_list.append(train_loss)
    ys_train = (y_true_train_list, y_prob_train_list)
    metrics_train = performances(y_true_train_list, y_prob_train_list, print_ = False)   
    return ys_train, f_mean(loss_train_list).cpu().detach().numpy(), metrics_train


if not os.path.isdir('./model_test'):
	os.mkdir('./model_test')
    
def train(model, train_loader, test_loader, loss, num_epochs, updater, cols):    
    metric_best, ep_best = 0, -1
    path_saver = './model_test/RNN_{}_{}.pkl'.format(cols[-3], cols[-2])
    for epoch in range(num_epochs):
        _, train_loss, _  = train_epoch(model, train_loader, loss, updater)
        _, test_loss, metrics_test  = evaluate_epoch(model, test_loader, loss)
    
        metrics_ep = metrics_test[1]
        if metrics_ep > metric_best:
            metric_best, ep_best = metrics_ep, epoch
            log_file.write('Saving model: Best epoch = {} | metrics_Best = {:.4f}'.format(ep_best, metric_best) + "\n")
            log_file.flush()
            print('Saving model: Best epoch = {} | metrics_Best = {:.4f}'.format(ep_best, metric_best))
            torch.save(model.eval().state_dict(), path_saver)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size, hidden_size, output_size = 1, 16, 1
batch_size = 1024

##多个模型
cols_list = [['patient','mutant_seq','rnaseq_TPM','mut_netchop_score_ct','TAP_score','mutant_rank','mutant_rank_PRIME','response_type'],
            ['patient','mutant_seq','rnaseq_TPM','mut_netchop_score_ct','TAP_score','mutant_rank_netMHCpan','mutant_rank_PRIME','response_type'],
            ['patient','mutant_seq','rnaseq_TPM','mut_netchop_score_ct','TAP_score','mutant_rank','BigMHC_IM','response_type'],
            ['patient','mutant_seq','rnaseq_TPM','mut_netchop_score_ct','TAP_score','mutant_rank_netMHCpan','BigMHC_IM','response_type']]


num_epochs = 50

for cols in cols_list:
    print('*******{}_{}_model start ********'.format(cols[-3], cols[-2]))
    model = RNN(input_size, hidden_size, output_size).to(device)
    updater = torch.optim.Adam(model.parameters())
    loss = nn.BCELoss()
    train_data,train_loader = makedata(type_ = 'train', cols = cols, batch_size = batch_size, shuffle_ = True)
    test_data,test_loader = makedata(type_ = 'test', cols = cols, batch_size = batch_size, shuffle_ = False)
    train(model, train_loader, test_loader, loss, num_epochs, updater, cols)


log_file.close()