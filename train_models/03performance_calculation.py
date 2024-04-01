# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 16:38:48 2023

@author: lijiajun
"""
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve,auc

RNN = pd.read_csv('./result/rank_RNN.csv')
ML_voting = pd.read_csv('./data/rank_ML_voting.csv')

def Top(patient, rank):
    df = pd.DataFrame({'patient':patient, 'rank':rank})
    df1 = df.groupby('patient').agg(list)['rank'].to_frame().reset_index()
    df1['top20'] = df1['rank'].apply(lambda x : len([j for j in x if j <= 20]))
    df1['top50'] = df1['rank'].apply(lambda x : len([j for j in x if j <= 50]))
    df1['top100'] = df1['rank'].apply(lambda x : len([j for j in x if j <= 100]))
    
    df1 = df1.sort_values(by='patient', ascending=True)
    nci = [sum(df1.loc[:12,'top20']),sum(df1.loc[:12,'top50']),sum(df1.loc[:12,'top100'])]
    hitide = [sum(df1.loc[13:21,'top20']),sum(df1.loc[13:21,'top50']),sum(df1.loc[13:21,'top100'])]
    tesla = [sum(df1.loc[22:,'top20']),sum(df1.loc[22:,'top50']),sum(df1.loc[22:,'top100'])]
    total = [sum(df1.loc[:,'top20']),sum(df1.loc[:,'top50']),sum(df1.loc[:,'top100'])]
    return [nci, hitide, tesla, total]


RNN_MP_num = Top(RNN['patient'], RNN['RNN_MP_rank'])
RNN_NP_num = Top(RNN['patient'], RNN['RNN_NP_rank'])
RNN_MB_num = Top(RNN['patient'], RNN['RNN_MB_rank'])
RNN_NB_num = Top(RNN['patient'], RNN['RNN_NB_rank'])
RNN_voting_num = Top(RNN['patient'], RNN['RNN_voting_rank'])

ML_voting_num = Top(ML_voting['patient'], ML_voting['ML_voting_rank'])


num_list = [ML_voting_num,RNN_MP_num,RNN_NP_num,RNN_MB_num,RNN_NB_num,RNN_voting_num]

num_file = open("./result/result_num.csv", "w")
num_file.write(','.join(['Top20','Top50','Top100','dataset','method',]))
num_file.write('\n')
method = ['ML_voting','RNN_MP','RNN_NP','RNN_MB','RNN_NB','RNN_voting']
for nums,method in zip(num_list,method):
    dataset = ['NCI-test', 'HiTIDE', 'TESLA', 'TOTAL']
    for n,dataset in zip(nums,dataset):
        n.append(dataset)
        n.append(method)
        num_file.write(','.join([str(i) for i in n]))
        num_file.write('\n')
num_file.close()        

def fr100(x):
    if x:
        return(len([j for j in x if j <= 100])/len(x))
    else:
        return 0

    
def ttif(x):
    if x:
        return(len([j for j in x if j <= 20])/20)
    else:
        return 0

def fun(x):
    l = [0] * max(x)
    for i in x:
        l[i-1]=1
    return l

def auprc(x):
    x = [j for j in x if j <= 100]
    if x:
        score = list(range(max(x),0,-1))
        lable = fun(x)
        precision, recall, _ = precision_recall_curve(lable,score)
        auprc = auc(recall,precision)
        return auprc
    else:
        return 0
        
def r_pod(x):
    if x:
        return((len([j for j in x if j <= 20])+sum([1/j for j in x if j <= 20]))/20)
    else:
        return 0
        

def cal(patient, rank):
    df = pd.DataFrame({'patient':patient, 'rank':rank})
    df1 = df.groupby('patient').agg(list)['rank'].to_frame().reset_index()
    df1['FR100'] = df1['rank'].apply(fr100)
    df1['TTIF'] = df1['rank'].apply(ttif)  
    df1['AUPRC'] = df1['rank'].apply(auprc)  
    df1['R_PoD'] = df1['rank'].apply(r_pod) 
    df1 = df1[['patient','FR100','TTIF','R_PoD','AUPRC']]
    return df1

RNN_MP= cal(RNN['patient'], RNN['RNN_MP_rank'])
RNN_NP = cal(RNN['patient'], RNN['RNN_NP_rank'])
RNN_MB = cal(RNN['patient'], RNN['RNN_MB_rank'])
RNN_NB = cal(RNN['patient'], RNN['RNN_NB_rank'])
RNN_voting = cal(RNN['patient'], RNN['RNN_voting_rank'].apply(int))

ML_voting = cal(ML_voting['patient'], ML_voting['ML_voting_rank'])

ls = ['NCI-test', 'HiTIDE', 'TESLA']
lt = [13,9,8]
dataset=[]
for i,n in zip(ls,lt):
    dataset=dataset+[i]*n


RNN_MP['dataset'] = dataset
RNN_NP['dataset'] = dataset
RNN_MB['dataset'] = dataset
RNN_NB['dataset'] = dataset
RNN_voting['dataset'] = dataset  
ML_voting['dataset'] = dataset 

RNN_MP['method'] = 'RNN_MP'
RNN_NP['method'] = 'RNN_NP'
RNN_MB['method'] = 'RNN_MB'
RNN_NB['method'] = 'RNN_NB'
RNN_voting['method'] = 'RNN_voting' 
ML_voting['method'] = 'ML_voting'

FR_TTIF_AUPRC = pd.concat([RNN_MP,RNN_NP,RNN_MB,RNN_NB,RNN_voting,ML_voting], axis = 0)

FR_TTIF_AUPRC.to_csv('./result/result_FR_TTIF_AUPRC_R-PoD.csv',index=False)



