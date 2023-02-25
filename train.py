import os

import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx

from utils import * 
from zc_metrics import * 
import pickle
import argparse
from torch import autograd
import torch.optim.lr_scheduler as lr_scheduler
from model import *

import matplotlib.pyplot as plt
import os
import tqdm
import random
# device = ("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()

#Model specific parameters
parser.add_argument('--input_size', type=int, default=64)
parser.add_argument('--output_size', type=int, default=5)
#parser.add_argument('--input_size_seg', type=int, default=12)
parser.add_argument('--input_size_seg', type=int, default=2)
parser.add_argument('--output_size_seg', type=int, default=5)
parser.add_argument('--n_stgcnn', type=int, default=1,help='Number of ST-GCNN layers')
parser.add_argument('--n_txpcnn', type=int, default=5, help='Number of TXPCNN layers')
parser.add_argument('--kernel_size', type=int, default=3)

#Data specifc paremeters
parser.add_argument('--obs_seq_len', type=int, default=4)  # observation
parser.add_argument('--pred_seq_len', type=int, default=12)  # prediction
parser.add_argument('--dataset', default='eth',help='eth,hotel,univ,zara1,zara2')
                    #help='eth,hotel,univ,zara1,zara2')    

#Training specifc parameters
parser.add_argument('--batch_size', type=int, default=1024,
                    help='minibatch size')
parser.add_argument('--num_epochs', type=int, default=50,
                    help='number of epochs')  
parser.add_argument('--clip_grad', type=float, default=None,
                    help='gadient clipping')        
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--lr_sh_rate', type=int, default=5,
                    help='number of steps to drop the lr')  
parser.add_argument('--use_lrschd', action="store_true", default=True,
                    help='Use lr rate scheduler')
parser.add_argument('--tag', default='tag', 
                    help='personal tag for the model ')                    
args = parser.parse_args()

print('*'*30)
print("Training initiating....")
print(args)

def graph_loss(V_pred,V_target):
    return bivariate_loss(V_pred,V_target)

#Data prep     
obs_seq_len = args.obs_seq_len
pred_seq_len = args.pred_seq_len
data_set = './datasets/'+args.dataset+'/'
data_set_train= './datasets/eth/'


batch_train = TrajectoryDataset(
        data_set+'train/',         #train，load training dataset
        obs_len=obs_seq_len,       #default=8
        pred_len=pred_seq_len,     #default=12
        skip=1,norm_lap_matr=True,
        type_='train')   #norm(v)    lap:Laplace matrix 



batch_val = TrajectoryDataset(
        data_set+'val/',         #train，load validation dataset
        obs_len=obs_seq_len,       #default=8
        pred_len=pred_seq_len,     #default=12
        skip=1,norm_lap_matr=True,
        type_='val')   #norm(v)   lap:Laplace matrix 



#Defining the model 
model = social_stgcnn(n_stgcnn =args.n_stgcnn,n_txpcnn=args.n_txpcnn,
input_feat=args.input_size,
output_feat=args.output_size,
input_feat_seg=args.input_size_seg,
output_feat_seg=args.output_size_seg,
seq_len=args.obs_seq_len,
pred_seq_len=args.pred_seq_len,kernel_size=args.kernel_size)
# model=model.to(device)
#Training settings 

optimizer = optim.Adam(model.parameters(),lr=args.lr)

if args.use_lrschd:
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_sh_rate, gamma=0.2)

checkpoint_dir = './checkpoint_zc_64_node+clu+ttc/'+args.tag+'/'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

with open(checkpoint_dir+'args.pkl', 'wb') as fp:
    pickle.dump(args, fp)
    


print('Data and model loaded')
print('Checkpoint dir:', checkpoint_dir)

#Training 
metrics = {'train_loss':[], 'val_loss':[]}
constant_metrics={'min_train_epoch':-1, 'min_train_loss':9999999999999999,'min_val_epoch':-1, 'min_val_loss':9999999999999999}


def train(epoch):
    global metrics,loader_train,constant_metrics

    model.train()
    loss_batch = 0 
    batch_count = 0
    is_fst_loss = True

    obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
    loss_mask,seq_start_end,sa,se,cluster,pedestrian_index,vehicle_index,rider_index= batch_train

#     obs_traj= obs_traj.to(device)
#     pred_traj_gt= pred_traj_gt.to(device)
#     obs_traj_rel=obs_traj_rel.to(device)
#     pred_traj_gt_rel=pred_traj_gt_rel.to(device)
#     non_linear_ped=non_linear_ped.to(device)
#     loss_mask=loss_mask.to(device)
#     sa=sa.to(device)
#     se=se.to(device)
#
# #    cluster = torch.stack(cluster,dim=1)
# #    torch.from_numpy(cluster)
#     cluster = torch.tensor(cluster)  # list
#     cluster=cluster.to(device)
#
#     pedestrian_index=torch.tensor(pedestrian_index) #list
#     pedestrian_index = pedestrian_index.to(device)
#
#     vehicle_index=torch.tensor(vehicle_index)   #list
#     vehicle_index = vehicle_index.to(device)
#
#     rider_index=torch.tensor(rider_index) #list
#     rider_index = rider_index.to(device)

    
    loader_len = len(seq_start_end)
    turn_point =int(loader_len/args.batch_size)*args.batch_size+ loader_len%args.batch_size -1
    obs_traj=np.squeeze(obs_traj,axis=0)
    pred_traj_gt=np.squeeze(pred_traj_gt,axis=0)
    obs_traj_rel=np.squeeze(obs_traj_rel,axis=0)
    pred_traj_gt_rel=np.squeeze(pred_traj_gt_rel,axis=0)
    
    sa=np.squeeze(sa,axis=0)
    se=np.squeeze(se,axis=0)
    
    seq_start_end = torch.tensor(seq_start_end)
    index = [i for i in range(len(seq_start_end))]
    random.shuffle(index)
    seq_start_end = seq_start_end[index]
    # seq_start_end = seq_start_end.to(device)

    batch_count = 0
    for ss in range(len(seq_start_end)):
        batch_count+=1
        cnt=ss
        start, end = seq_start_end[ss]
        obs_len=obs_seq_len
        se_out=se[ss,0:obs_len,:]
        sa_out=sa[ss,0:obs_len,:]
        norm_lap_matr=True
        v_tr,a_=seq_to_graph(pred_traj_gt[start:end,:],pred_traj_gt_rel[start:end, :],norm_lap_matr)
        optimizer.zero_grad()

        V_pred,_,_=model(cluster,obs_traj,obs_traj_rel,pred_traj_gt,start,pred_traj_gt_rel,end,sa_out,se_out,pedestrian_index,vehicle_index,rider_index)
        V_pred = V_pred.permute(0,2,3,1)
        V_pred = V_pred.squeeze()
        V_tr = v_tr
        if batch_count%args.batch_size !=0 and cnt != turn_point :
            l = graph_loss(V_pred,V_tr)
            if is_fst_loss :
                loss = l
                is_fst_loss = False
            else:
                loss += l

        else:
            loss = loss/args.batch_size
            print('loss',loss)
            is_fst_loss = True
            loss.backward()
                 
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad)

            optimizer.step()
            #Metrics
            loss_batch += loss.item()
            print('TRAIN:','\t Epoch:', epoch,'\t Loss:',loss_batch/batch_count)
                    
    metrics['train_loss'].append(loss_batch/batch_count)
    fig = plt.figure(figsize=(6, 4))        
    plt.plot(range(1, len(metrics['train_loss']) + 1), metrics['train_loss'], label='Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.xlim(0, len(metrics['train_loss']) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.title( " train loss")
    fig.savefig(checkpoint_dir+'train_loss.png', bbox_inches='tight')
    if  metrics['train_loss'][-1] < constant_metrics['min_train_loss']:
        constant_metrics['min_train_loss'] =  metrics['train_loss'][-1]
        constant_metrics['min_train_epoch'] = epoch
        torch.save(model.state_dict(),checkpoint_dir+'train_best.pth')  # OK
        
        
    
    


def vald(epoch):
    global metrics,loader,constant_metrics
    model.eval()
    loss_batch = 0 
    batch_count = 0
    is_fst_loss = True

    obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
    loss_mask,seq_start_end,sa,se,cluster,pedestrian_index,vehicle_index,rider_index= batch_val
    
    loader_len = len(seq_start_end)
    turn_point =int(loader_len/args.batch_size)*args.batch_size+ loader_len%args.batch_size -1
    obs_traj=np.squeeze(obs_traj,axis=0)
    pred_traj_gt=np.squeeze(pred_traj_gt,axis=0)
    obs_traj_rel=np.squeeze(obs_traj_rel,axis=0)
    pred_traj_gt_rel=np.squeeze(pred_traj_gt_rel,axis=0)
    sa=np.squeeze(sa,axis=0)
    se=np.squeeze(se,axis=0)
    batch_count = 0

    for ss in range(len(seq_start_end)):
        batch_count+=1
        cnt=ss
        start, end = seq_start_end[ss]
        obs_len=obs_seq_len
        se_out=se[ss,0:obs_len,:]
        sa_out=sa[ss,0:obs_len,:]
        norm_lap_matr=True
        v_tr,a_=seq_to_graph(pred_traj_gt[start:end,:],pred_traj_gt_rel[start:end, :],norm_lap_matr)
        optimizer.zero_grad()
        V_pred,_,_=model(cluster,obs_traj,obs_traj_rel,pred_traj_gt,start,pred_traj_gt_rel,end,sa_out,se_out,pedestrian_index,vehicle_index,rider_index)
        V_pred = V_pred.permute(0,2,3,1)
        V_pred = V_pred.squeeze()
        V_tr = v_tr
         
        if batch_count%args.batch_size !=0 and cnt != turn_point: #args.batch_size=128
            l = graph_loss(V_pred,V_tr)            
            if is_fst_loss :
                loss = l
                is_fst_loss = False
            else:
                loss += l
        else:
            loss = loss/args.batch_size
            is_fst_loss = True
            #Metrics 
            loss_batch += loss.item()
            print('VALD:','\t Epoch:', epoch,'\t Loss:',loss_batch/batch_count)
    metrics['val_loss'].append(loss_batch/batch_count)
    fig = plt.figure(figsize=(6, 4))        
    plt.plot(range(1, len(metrics['val_loss']) + 1), metrics['val_loss'], label='Loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.xlim(0, len(metrics['val_loss']) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.title( "val loss")

    fig.savefig(checkpoint_dir+'val_loss.png', bbox_inches='tight')

    if  metrics['val_loss'][-1] < constant_metrics['min_val_loss']:
        constant_metrics['min_val_loss'] =  metrics['val_loss'][-1]
        constant_metrics['min_val_epoch'] = epoch
        torch.save(model.state_dict(),checkpoint_dir+'val_best.pth')  # OK
    
    
for epoch in range(args.num_epochs):
    train(epoch)
    vald(epoch)
    if args.use_lrschd:
        scheduler.step()


    print('*'*30)
    print('Epoch:',args.tag,":", epoch)

    print(constant_metrics)
    print('*'*30)
    
    with open(checkpoint_dir+'metrics.pkl', 'wb') as fp:
        pickle.dump(metrics, fp)
    
    with open(checkpoint_dir+'constant_metrics.pkl', 'wb') as fp:
        pickle.dump(constant_metrics, fp)  













