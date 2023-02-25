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


 def ade(predAll,targetAll,count_):   
     All = len(predAll)
     sum_all = 0
     maxade = []
     tmpde = 0
     for s in range(All):
         pred = np.swapaxes(predAll[s][:,:count_[s],:],0,1)
         target = np.swapaxes(targetAll[s][:,:count_[s],:],0,1)
         N = pred.shape[0]
         T = pred.shape[1]
         sum_ = 0
         for i in range(N):
             for t in range(T):
                 tmpde = math.sqrt((pred[i,t,0] - target[i,t,0])**2+(pred[i,t,1] - target[i,t,1])**2)
         sum_ = tmpde
         sum_all += sum_/(N*T)
         maxade = maxade.append(tmpde)
     return sum_all/All,max(maxade)

def minade_index(predAll, targetAll, count_):
    All = len(predAll)
    sum_all = 0
    for s in range(All):
        pred = np.swapaxes(predAll[s][:, :count_[s], :], 0, 1)
        target = np.swapaxes(targetAll[s][:, :count_[s], :], 0, 1)

        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0
        tmpade = 0
        ade_N = []
        for i in range(N):
            for t in range(T):
                tmpade == math.sqrt((pred[i, t, 0] - target[i, t, 0]) ** 2 + (pred[i, t, 1] - target[i, t, 1]) ** 2)
                sum_ += tmpade
                ade_N.append(tmpade)
        sum_all += sum_ / (N * T)

        # obtain the trajectory index with minADE
        labelindex_set = (np.array(ade_N) == min(ade_N))
        listlabel = list(labelindex_set)
        index_ade = listlabel.index(True)
        # predicted trajectory with minADE
        min_ade_result_x = pred[index_ade, :, 0]
        min_ade_result_y = pred[index_ade, :, 1]
        min_ade_result = [min_ade_result_x, min_ade_result_y]
        # gt trajectory
        gt_result_x = target[index_ade, :, 0]
        gt_result_y = target[index_ade, :, 0]
        gt_result = [gt_result_x, gt_result_y]

    return sum_all / All, gt_result, min_ade_result



def ade(predAll,targetAll,count_):
    All = len(predAll)
    sum_all = 0
    for s in range(All):
        pred = np.swapaxes(predAll[s][:,:count_[s],:],0,1)
        target = np.swapaxes(targetAll[s][:,:count_[s],:],0,1)
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0
        for i in range(N):
            for t in range(T):
                sum_+=math.sqrt((pred[i,t,0] - target[i,t,0])**2+(pred[i,t,1] - target[i,t,1])**2)
        sum_all += sum_/(N*T)
    return sum_all/All

def fde(predAll,targetAll,count_):
    All = len(predAll)
    sum_all = 0 
    for s in range(All):
        pred = np.swapaxes(predAll[s][:,:count_[s],:],0,1)
        target = np.swapaxes(targetAll[s][:,:count_[s],:],0,1)
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0 
        for i in range(N):
            for t in range(T-1,T):
                sum_+=math.sqrt((pred[i,t,0] - target[i,t,0])**2+(pred[i,t,1] - target[i,t,1])**2)
        sum_all += sum_/(N)
    return sum_all/All



def fde_frame(predAll,targetAll,count_):
    All = len(predAll)
    sum_all = 0 
    for s in range(All):
        pred = predAll[s][:count_[s],:]
        target = targetAll[s][:count_[s],:]
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0 
        for i in range(N):
            sum_+=math.sqrt((pred[i,0] - target[i,0])**2+(pred[i,1] - target[i,1])**2)
        sum_all += sum_/(N)
    return sum_all/All

def seq_to_nodes(seq_,max_nodes = 88):   #Convert the input sequence to nodes, the parameter max_nodes indicates the maximum number of nodes, the default value is 88
    seq_ = seq_.squeeze()
    seq_len = seq_.shape[2]
    
    V = np.zeros((seq_len,max_nodes,2))    #By mapping each step in the sequence to the maximum number of nodes, and then saving the nodes in V
    for s in range(seq_len):
        step_ = seq_[:,:,s]
        for h in range(len(step_)): 
            V[s,h,:] = step_[h]
    return V.squeeze()


def nodes_rel_to_nodes_abs_frame(nodes,init_node):  #Convert relative nodes to absolute nodes, nodes indicates the relative nodes to be converted, and init_node indicates the initial node
    nodes_ = np.zeros_like(nodes)

    for ped in range(nodes.shape[0]):
        nodes_[ped,:] = nodes[ped][:] + init_node[ped][:]     #Add relative nodes to initial nodes to get absolute nodes and save them in nodes_[ped,:]
    
    return nodes_.squeeze()

def seq_to_nodes(seq_,max_nodes = 88):
    seq_ = seq_.squeeze()
    seq_len = seq_.shape[2]
    
    V = np.zeros((seq_len,max_nodes,2))
    for s in range(seq_len):
        step_ = seq_[:,:,s]
        for h in range(len(step_)): 
            V[s,h,:] = step_[h]
            
    return V.squeeze()
#
def nodes_rel_to_nodes_abs(nodes,init_node):
    nodes_ = np.zeros_like(nodes)
    for s in range(nodes.shape[0]):
        for ped in range(nodes.shape[1]):
            nodes_[s,ped,:] = np.sum(nodes[:s+1,ped,:],axis=0) + init_node[ped,:]

    return nodes_.squeeze()

def closer_to_zero(current,new_v):
    dec =  min([(abs(current),current),(abs(new_v),new_v)])[1]
    if dec != current:
        return True
    else: 
        return False
        
def bivariate_loss(V_pred,V_trgt):      #Calculate the bivariate loss of V_pred predicted value, V_trgt target value
    normx = V_trgt[:,:,0]- V_pred[:,:,0]   #Normx, normy represent the deviation between the target value and the predicted value
    normy = V_trgt[:,:,1]- V_pred[:,:,1]

    sx = torch.exp(V_pred[:,:,2]) #sx，sy
    sy = torch.exp(V_pred[:,:,3])
    corr = torch.tanh(V_pred[:,:,4]) #corr

    sxsy = sx * sy

    z = (normx/sx)**2 + (normy/sy)**2 - 2*((corr*normx*normy)/sxsy)
    negRho = 1 - corr**2
    for i in range(negRho.shape[0]):
        for j in range(negRho.shape[1]):
            if negRho[i,j]==0.0000:
                negRho[i,j]=1e-10
    result = torch.exp(-z/(2*negRho))
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))
    for i in range(denom.shape[0]):
        for j in range(denom.shape[1]):
            if denom[i,j]==0.0000:
                denom[i,j]=1e-10
    result = result / denom
    epsilon = 1e-20
    result = -torch.log(torch.clamp(result, min=epsilon))
    result = torch.mean(result)
    return result
   
