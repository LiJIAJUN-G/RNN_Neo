# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 11:43:27 2023

@author: lijiajun
"""

import torch
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from torch.utils.data import DataLoader
from torch import nn
from sklearn.metrics import roc_curve, auc, precision_recall_curve



def makedata(type_, cols,  batch_size, shuffle_):
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size, hidden_size, output_size = 1, 16, 1
batch_size = 1024


cols_list = [['patient','mutant_seq','rnaseq_TPM','mut_netchop_score_ct','TAP_score','mutant_rank','mutant_rank_PRIME','response_type'],
            ['patient','mutant_seq','rnaseq_TPM','mut_netchop_score_ct','TAP_score','mutant_rank_netMHCpan','mutant_rank_PRIME','response_type'],
            ['patient','mutant_seq','rnaseq_TPM','mut_netchop_score_ct','TAP_score','mutant_rank','BigMHC_IM','response_type'],
            ['patient','mutant_seq','rnaseq_TPM','mut_netchop_score_ct','TAP_score','mutant_rank_netMHCpan','BigMHC_IM','response_type']]



def get_per_patient_rank(y_true, y_prob, patient, pep, save_ = True):
    re={'y_true':y_true, 'y_prob':y_prob, 'patient':patient, 'pep':pep}
    df = pd.DataFrame(re)
    sorted_groups = df.sort_values(by=['patient','y_prob'],ascending=False)
    sorted_groups['rank'] = sorted_groups.groupby('patient').cumcount() + 1
    if save_:
        result = sorted_groups[sorted_groups['y_true'] == 1]
        result.to_csv('RNN_result.csv',index=False)
    return sorted_groups


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size, hidden_size, output_size = 1, 16, 1
batch_size = 1024

def bind_all_model(model_path, cols_list):
    df = pd.DataFrame()
    for index, cols in enumerate(cols_list):
        for file in os.listdir(model_path):
            if file.endswith('{}_{}.pkl'.format(cols[-3], cols[-2])):
                print('****正在处理'+file)
                model = RNN(input_size, hidden_size, output_size).to(device)
                loss = nn.BCELoss()
                model.load_state_dict(torch.load(model_path+'/'+file), strict = True)
                test_data,test_loader = makedata(type_ = 'test', cols = cols,  batch_size = batch_size, shuffle_ = False)
                ys_val,_ ,_  = evaluate_epoch(model, test_loader, loss)
                result = get_per_patient_rank(np.array(ys_val[0]).flatten(),np.array(ys_val[1]).flatten(),test_data.loc[:,'patient'],test_data.loc[:,'mutant_seq'],save_ = False)
                result.rename(columns={'rank': '{}_{}_rank'.format(cols[-3], cols[-2]),
                                      'y_prob': '{}_{}_prob'.format(cols[-3], cols[-2])},inplace=True)
                result.sort_values(by='{}_{}_rank'.format(cols[-3], cols[-2]) , inplace=True, ascending=True) 
                if df.empty:
            
                    df = pd.concat([df,result],axis=1)
                else:
                    df = df.merge(result,how='left',on=['y_true','pep','patient']) 
    return df



model_path = './model'
muti_RNN = bind_all_model(model_path, cols_list)


def sum_rank(df):
    selected_columns = [col for col in df.columns if col.endswith('prob')]
    print(selected_columns)
    df['sum'] = df[selected_columns].apply( lambda x: (x - x.mean()) / x.std(), axis=0).sum(axis=1)
    df['sum_rank'] = df.groupby(['patient'])['sum'].rank(method='min',ascending=False)
    df = df[df['y_true'] == 1]
    return df


re = sum_rank(muti_RNN)

re = re[['pep','patient','mutant_rank_mutant_rank_PRIME_prob','mutant_rank_mutant_rank_PRIME_rank','mutant_rank_netMHCpan_mutant_rank_PRIME_prob','mutant_rank_netMHCpan_mutant_rank_PRIME_rank','mutant_rank_BigMHC_IM_prob','mutant_rank_BigMHC_IM_rank','mutant_rank_netMHCpan_BigMHC_IM_prob','mutant_rank_netMHCpan_BigMHC_IM_rank','sum','sum_rank']]
re.columns = ['pep','patient','RNN_MP_prob','RNN_MP_rank','RNN_NP_prob','RNN_NP_rank','RNN_MB_prob','RNN_MB_rank','RNN_NB_prob','RNN_NB_rank','RNN_voting','RNN_voting_rank']
re.to_csv('./result/rank_RNN.csv', index=False)


