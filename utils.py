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
#import networkx as nx
from tqdm import tqdm
import time
import pandas as pd
from dtw import dtw
from numpy.linalg import norm
import numpy as np
from sklearn.cluster import SpectralClustering
import re
import random
import json
import pandas as pd

def anorm(p1,p2): 
    NORM = math.sqrt((p1[0]-p2[0])**2+ (p1[1]-p2[1])**2)
    if NORM ==0:
        return 0
    return 1/(NORM)



def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)   #np.linspace主要用来创建等差数列
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0
    
    
def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
#    if delim == ' ':
#        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [i for i in line]
            data.append(line)
    return np.asarray(data)

def custom_norm(x, y):
    temp=np.array(x)-np.array(y)
    ant=norm(temp)
    distance=math.sqrt(2*(1-math.exp(-(ant*ant)/0.5)))
    return distance

class DtwCluster():
    def __init__(self, data):
        self.data = data
        self.data_len = len(data)
        self.dtw_distance_matrix = np.zeros((self.data_len, self.data_len))
        

    def cal_dis_matrix(self):
        for i in range(self.data_len):
            for j in range(self.data_len):
                self.dtw_distance_matrix[i][j] = self.cal_dtw_distance(self.data[i], self.data[j])
        return self.dtw_distance_matrix
    
    def clustering_k(self, num_clusters):
        clusters = SpectralClustering(n_clusters=num_clusters, affinity='precomputed').fit(self.dtw_distance_matrix)
        return clusters.labels_
    
    @staticmethod
    def cal_dtw_distance(t1, t2):
#        dist, cost, acc, path = dtw(t1, t2, dist=custom_norm)
        dist=custom_norm(t1,t2)
        return dist
    
def class_cluster(seq_traj_xytype_zc,type_):
    class_clu=np.zeros(len(seq_traj_xytype_zc))

    pedestrian=[]
    pedestrian_index=[]
    vehicle=[]
    vehicle_index=[]
    rider=[]
    rider_index=[]
    for i in range(0,len(seq_traj_xytype_zc)):

        if seq_traj_xytype_zc[i,0,-1]==0:
            pedestrian.append(seq_traj_xytype_zc[i,:,:2])
            pedestrian_index.append(i)

        if seq_traj_xytype_zc[i,0,-1]==1:
            vehicle.append(seq_traj_xytype_zc[i,:,:2])
            vehicle_index.append(i)

        if seq_traj_xytype_zc[i,0,-1]==2:
            rider.append(seq_traj_xytype_zc[i,:,:2])
            rider_index.append(i)
    print('pedestrian',len(pedestrian))
    print('vehicle',len(vehicle))
    print('rider',len(rider))

    
    if type_=='train':
        df = pd.read_csv(open(r'D:\risg\risg程序\DALUNWEN - gai-2\clu\train_clu.csv'))
    if type_=='val':
        df = pd.read_csv(open(r'D:\risg\risg程序\DALUNWEN - gai-2\clu\val_clu.csv'))
    if type_=='test':
        df = pd.read_csv(open(r'D:\risg\risg程序\DALUNWEN - gai-2\clu\test_clu.csv'))
    # df = pd.read_csv(open(r'D:\risg\risg程序\DALUNWEN - gai-2\clu\test_clu.csv'))
    class_clu=df['clu'].tolist()
#    print('class_clu',class_clu)

#    if len(pedestrian)==1:
#        clu0 = [1]
#    else:
#        clustering0 = DtwCluster(pedestrian)
#        clustering0.cal_dis_matrix()
#        clusters0= clustering0.clustering_k(6)
#        clu0=clusters0+1
#    print('clu0',clu0)
#    clustering1 = DtwCluster(vehicle)
#    clustering1.cal_dis_matrix()
#    clusters1 = clustering1.clustering_k(3)
#    clu1=clusters1+1
#    print('clu1',clu1)
#    if len(rider)!=0:
#        if len(rider)<3:
#            clustering2 = DtwCluster(rider)
#            clustering2.cal_dis_matrix()
#            clusters2 = clustering2.clustering_k(1)
#            clu2 = clusters2+1 
#        else:
#            clustering2 = DtwCluster(rider)
#            clustering2.cal_dis_matrix()
#            clusters2 = clustering2.clustering_k(3)
#            clu2 = clusters2+1
#        print('clu2',clu2)
#    
#    for ii,c0 in enumerate(pedestrian_index):
#        class_clu[c0]=clu0[ii]
#       
#    for jj,c1 in enumerate(vehicle_index):
#        class_clu[c1]=clu1[jj]
#       
#    if len(rider)!=0:
#        for kk,c2 in enumerate(rider_index):
#            class_clu[c2]=clu2[kk]
##    print('class_clu',class_clu)
#    
#    c=pd.DataFrame()
#    c['clu']=class_clu
#    #print('c',c)
##    c['index']=class_clu_index
##    print('c',c)
#    if type_=='train':
#        c.to_csv(r'D:\risg\risg程序\DALUNWEN - gai-2\clu\train_clu.csv')
#    if type_=='val':
#        c.to_csv(r'D:\risg\risg程序\DALUNWEN - gai-2\clu\val_clu.csv')
#    if type_=='test':
#        c.to_csv(r'D:\risg\risg程序\DALUNWEN - gai-2\clu\test_clu.csv')
    return class_clu,pedestrian_index,vehicle_index,rider_index
                         
                         
                            
def padding_se(graph):
    length=[]
    for i in range(len(graph)):
        d=len(graph[i][0])
        length.append(d)
    max_length=max(length)

    for j in range(len(graph)):
        if len(graph[j][0])<max_length:
            l=max_length-len(graph[j][0])
            b=[0]*l
            for jj in range(len(graph[j])):
                graph[j][jj]=graph[j][jj]+b

        if len(graph[j])<max_length:
            l2=max_length-len(graph[j])
            c=[0]*max_length
            c1=[c]*l2
            graph[j]=np.row_stack((graph[j], c1))
    return graph
            
def padding_sa(graph):
    length=[]

    for i in range(len(graph)):
        d=len(graph[i])
        length.append(d)
    max_length=max(length)

    for j in range(len(graph)):
        if len(graph[j])<max_length:
            l2=max_length-len(graph[j])
            c=[0]*12
            c1=[c]*l2
            
            for jj in range(len(graph[j])):
                if len(graph[j][jj])>12:
                    graph[j][jj]=graph[j][jj][:12]
            graph[j]=np.row_stack((graph[j], c1))
    return graph



def TrajectoryDataset(
        data_dir, obs_len=5, pred_len=8, skip=1, threshold=0.002,
        min_ped=1, delim='\t',norm_lap_matr = True,type_='train'):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """

        max_peds_in_frame = 0
        data_dir = data_dir
        obs_len = obs_len
        pred_len = pred_len
        skip = skip
        seq_len = obs_len + pred_len
        delim = delim
        norm_lap_matr = norm_lap_matr
        
        all_files = os.listdir(data_dir)
        all_files = [os.path.join(data_dir, _path) for _path in all_files]
        
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        seq_list_xytype_zc=[]
        se22=[]
        sa22=[]
        with open(r'D:\risg\risg程序\DALUNWEN - gai-2\datasets\segment\train\seg_ment_trainval_global.json', 'r',
                  encoding='utf8')as fp:
            json_data = json.load(fp)
        
        for path in all_files:
            data = read_file(path, delim)
            u, ind = np.unique((data[:, 0]), return_index=True)
            frames=u[np.argsort(ind)].tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])     #len(frame_data)=697张图片
            print('frame_data',len(frame_data))
            se1=[]
            sa1=[]
            for i in range(len(frame_data)):
                se=[json_data[j]['se'] for j in range(len(json_data)) if json_data[j]['sample_token']==frame_data[i][0][0]]
                if len(se)!=0:
                    if len(se[0])==0 or len(se[0][0])==0 or len(se[0])==1 or len(se[0][0])==1:
                        se[0]=[[0,0],[0,0]]
                    se1.append(se[0])
                sa=[json_data[j]['sa'] for j in range(len(json_data)) if json_data[j]['sample_token']==frame_data[i][0][0]]

                if len(sa)!=0:
                    if len(sa[0])==0 or len(sa[0][0])==0 or len(sa[0])==1 or len(sa[0][0])==1:
                        sa[0]=[[0]*12,[0]*12]
                    sa1.append(sa[0])
            
            se1=padding_se(se1)
            sa1=padding_sa(sa1)       

            num_sequences = int(
                math.ceil((len(frames) - seq_len + 1) / skip))
            
            for idx in range(0, num_sequences * skip + 1, skip):

                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + seq_len], axis=0)

                u1, ind1 = np.unique((curr_seq_data[:, 1]), return_index=True)
                peds_in_curr_seq=u1[np.argsort(ind1)]
                max_peds_in_frame = max(max_peds_in_frame,len(peds_in_curr_seq))
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                         seq_len))
                curr_seq1 = np.zeros((len(peds_in_curr_seq), 2, seq_len))
                curr_seq_xytype_zc = np.zeros((len(peds_in_curr_seq), 3, seq_len))
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                          seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                
                for _, ped_id in enumerate(peds_in_curr_seq):

                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]

                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx

                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1


                    if pad_end - pad_front != len(curr_ped_seq) or pad_end - pad_front != seq_len:
                        continue
                    
                    b=[]
                    for k in range(len(curr_ped_seq)):
                        b.append([float(i) for i in curr_ped_seq[:,2:][k]])
                    curr_ped_seq_xytype = np.around(b, decimals=4)
                    curr_ped_seq1 = np.transpose(curr_ped_seq_xytype[:,:2])
                    curr_ped_seq_xytype=np.transpose(curr_ped_seq_xytype)
                    curr_ped_seq=np.transpose(curr_ped_seq)
                    curr_ped_seq1 = curr_ped_seq1
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq1.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq1[:, 1:] - curr_ped_seq1[:, :-1]
                    _idx = num_peds_considered

                    curr_seq1[_idx, :, pad_front:pad_end] = curr_ped_seq1
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    curr_seq_xytype_zc[_idx, :, pad_front:pad_end] = curr_ped_seq_xytype

                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(
                        poly_fit(curr_ped_seq1, pred_len, threshold))
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1
                    se2=se1[pad_front:pad_end]
                    sa2=sa1[pad_front:pad_end]
                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq1[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                    seq_list_xytype_zc.append(curr_seq_xytype_zc[:num_peds_considered])

                    se22.append(se2)
                    sa22.append(sa2)
        num_seq = len(seq_list)
        print('self.num_seq',num_seq)
        seq_list = np.concatenate(seq_list, axis=0)
        print('len(seq_list)',len(seq_list))

        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)


        seq_list_xytype_zc = np.concatenate(seq_list_xytype_zc,axis=0)
        seq_traj_xytype_zc=np.transpose(seq_list_xytype_zc,(0,2,1))
        print('seq_traj_xytype_zc',seq_traj_xytype_zc.shape)

        se22=torch.Tensor(se22)
        print('se22.shape',se22.shape)

        sa22=torch.Tensor(sa22)
        print('sa22.shape',sa22.shape)
            
            
        class_cluster_zc,pedestrian_index,vehicle_index,rider_index=class_cluster(seq_traj_xytype_zc,type_)

        # Convert numpy -> Torch Tensor
        obs_traj = torch.from_numpy(
            seq_list[:, :, :obs_len]).type(torch.float)
        print('self.obs_traj.shape',obs_traj.shape)
      
        pred_traj = torch.from_numpy(
            seq_list[:, :, obs_len:]).type(torch.float)
        obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :obs_len]).type(torch.float)
        pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, obs_len:]).type(torch.float)
        loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        
        out = [
            obs_traj, pred_traj,
            obs_traj_rel, pred_traj_rel,
            non_linear_ped, loss_mask,
            seq_start_end,
            sa22, se22,
            class_cluster_zc,
            pedestrian_index,
            vehicle_index,
            rider_index
        ]
        return out
        
