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
import networkx as nx
import torch.optim as optim

def anorm(p1,p2):    #compute the Euclidean Distance
    NORM = math.sqrt((p1[0]-p2[0])**2+ (p1[1]-p2[1])**2)
    if NORM ==0:
        return 0
    return 1/(NORM)


def seq_to_graph(seq_,seq_rel,norm_lap_matr = True):   # graph construction
    seq_ = seq_.squeeze()             
    seq_rel = seq_rel.squeee()
    seq_len = seq_.shape[2]           # seq length
    max_nodes = seq_.shape[0]         # graph nodes
    
    V = np.zeros((seq_len,max_nodes,2))  # 8，3,2 or 12,3,2
    A = np.zeros((seq_len,max_nodes,max_nodes))
    for s in range(seq_len):
        step_ = seq_[:,:,s]           # 
        step_rel = seq_rel[:,:,s]     # 
        for h in range(len(step_)): 
            V[s,h,:] = step_rel[h]
            A[s,h,h] = 1
            for k in range(h+1,len(step_)):
                l2_norm = anorm(step_rel[h],step_rel[k])
                A[s,h,k] = l2_norm
                A[s,k,h] = l2_norm
        if norm_lap_matr:     # 
            G = nx.from_numpy_matrix(A[s,:,:])
            A[s,:,:] = nx.normalized_laplacian_matrix(G).toarray()

    return torch.from_numpy(V).type(torch.float),\
           torch.from_numpy(A).type(torch.float)

class MLP(nn.Module):    # mlp
    def __init__(self,inputsize,commonsize):    # inputsize,commonsize
        super(MLP,self).__init__()
        self.linear=nn.Sequential(               # self.linear，3 layers
                nn.Linear(inputsize,128),        # 
                nn.PReLU(),                      # 
                nn.Linear(128,64),               # 
                nn.PReLU(),                      # 
                nn.Linear(64,commonsize),        # 
                nn.PReLU()
                )
    def forward(self,x):
        out=self.linear(x)
        return out 

def angle_l(a):     #compute the angle of agents
    x1= a[-1][0]-a[-2][0]
    y1= a[-1][1]-a[-2][1]
    if x1==0:
        angle1=np.pi/2
    else:
        angle1=math.atan(y1/x1)
    return angle1


class node_o(nn.Module):    #compute the MLP feature for agents
    def __init__(self):
        super(node_o,self).__init__()

#        self.mlp1 = MLP(4,1)
        
    def forward(self,a):      # a:location
        node=[]
        node_64=[]
        for q in range(1,a.shape[0]):
            node_single=[]      # location buffer
            node_single_64=[]   # node dimension:64
            for qq in range(a[q].shape[0]):
                for qqq in range(a[q].shape[0]):
                    dis=np.sqrt((a[q][qq][0]-a[q][qqq][0])*(a[q][qq][0]-a[q][qqq][0])+(a[q][qq][1]-a[q][qqq][1])*(a[q][qq][1]-a[q][qqq][1]))    #dis：the distance of two agents
                    d1=np.sqrt((a[q][qq][0]-a[q-1][qq][0])*(a[q][qq][0]-a[q-1][qq][0])+(a[q][qq][1]-a[q-1][qq][1])*(a[q][qq][1]-a[q-1][qq][1]))
                    v1=d1/0.5
                    d2=np.sqrt((a[q][qqq][0]-a[q-1][qqq][0])*(a[q][qqq][0]-a[q-1][qqq][0])+(a[q][qqq][1]-a[q-1][qqq][1])*(a[q][qqq][1]-a[q-1][qqq][1]))  #d1,d2 the distance of agents in t-1,t
                    v2=d2/0.5                # V1,V2 velocity
                    angle1=angle_l([a[q-1][qq],a[q][qq]])
                    angle2=angle_l([a[q-1][qqq],a[q][qqq]])     # angle1, angle2

                    x_linju=[a[q][qq][0]]     # x_linju，y_linju：neighorhood agents
                    y_linju=[a[q][qq][1]]
                    v_linju=[v1]              # v_linju：velocity of neighorhood agents
                    angle_linju=[angle1]      # v_linju：angle of neighorhood agents
                    if dis<=12:
                        x_linju.append(a[q][qqq][0])
                        y_linju.append(a[q][qqq][1])
                        v_linju.append(v2)
                        angle_linju.append(angle2)

                    mlp1=MLP(len(x_linju),1)
                    mlp2=MLP(len(x_linju),1)
                    mlp3=MLP(len(x_linju),1)
                    mlp4=MLP(len(x_linju),1)
                    mlp5=MLP(4,1)
                    mlp6=MLP(4,64)           #1-6 mlp,predict the agent trajectories
                   
                    x_mlp=mlp1(torch.Tensor(x_linju))
                    x_mlp=x_mlp.detach().numpy().astype(float)
                    y_mlp=mlp2(torch.Tensor(y_linju))
                    y_mlp=y_mlp.detach().numpy().astype(float)
                    v_mlp=mlp3(torch.Tensor(v_linju))
                    v_mlp=v_mlp.detach().numpy().astype(float)
                    angle_mlp=mlp4(torch.Tensor(angle_linju))
                    angle_mlp=angle_mlp.detach().numpy().astype(float)
                    sss=mlp5(torch.Tensor([x_mlp[0],y_mlp[0],v_mlp[0],angle_mlp[0]]))   #location of predicted agents
                    sss_64=mlp6(torch.Tensor([x_mlp[0],y_mlp[0],v_mlp[0],angle_mlp[0]]))   #64-dimension of predicted agents
                    sss_64=sss_64.tolist()
                node_single.append(sss)
                node_single_64.append(sss_64)
            node.append(node_single)
            node_64.append(node_single_64)

        node_out=[node[0]]
        for jj in node:
            node_out.append(jj)       #jj for traversing the nodes and save the last one to node_out
        node_out2=torch.tensor(node_out)    #The node_out list is converted into a tensor and named node_out2
        #node_out1=np.array(node_out)
        #node_out1 = node_out1.astype(float)
        #node_out2=torch.from_numpy(node_out1).type(torch.float)
        node_out_64=[node_64[0]]
        for jj in node_64:
            node_out_64.append(jj)        #Traverse the elements in node_64 and add them to the node_out_64 list
        node_out2_64=torch.tensor(node_out_64)    #The ode_out_64 list is converted into a tensor and named node_out2_64
        return node_out2,node_out2_64


class risk_interaction(nn.Module):      #risk graph and scene graph
    def __init__(self):
        super(risk_interaction,self).__init__()
        self.node_o=node_o()
#        self.mlp = MLP(2,1)
        self.mlp = MLP(4,1)
        
    def forward(self,cluster,a,start,end,sa_out,se_out,pedestrian_index,vehicle_index,rider_index):    #Calculate risk interaction: cluster node cluster; a position information; start and end position of start and end node clusters; sa_out edge feature; 3 indexes represent the indexes of different entities of automobiles, bicycles and pedestrians；

        a=a.permute(2,0,1)
       
        clu=cluster[start:end]
        scene_graph_a=torch.cat((a, sa_out[:,:,-2:]), 1)   #a position information and sa_out edge feature splicing process is scene_graph_a/e
        scene_graph_e=torch.zeros((scene_graph_a.shape[0],scene_graph_a.shape[1],scene_graph_a.shape[1]))
        for p in range(se_out.shape[0]):
            for pp in range(se_out.shape[1]):
                for ppp in range(se_out.shape[1]):
                    scene_graph_e[p][pp][ppp]=se_out[p][pp][ppp]

        ped_ii=[]   # Extract indices of pedestrian_index and vehicle_index and store in ped_ii, veh_ii
        veh_ii=[]
        
        for ii,ind in enumerate(range(start,end)):
            if ind in pedestrian_index:
                ped_ii.append(ii)
            else:
                veh_ii.append(ii)
        node_ou,node_ou_64=self.node_o(a)   #Use node_0() and mlp(4,1) functions to process a to get node_ou, node_ou_64
        risk_inter=[]
        
        for k in range(1,a.shape[0]):
           for kk in range(sa_out.shape[1]):
               for e in ped_ii:
                   if sa_out[k][kk][4]==1.0 or sa_out[k][kk][5]==1.0:
                       scene_graph_e[k][kk][kk+e]=scene_graph_e[k][kk][kk+e]=1
               for ee in ped_ii:
                   if sa_out[k][kk][0]==1.0 or sa_out[k][kk][1]==1.0:
                       scene_graph_e[k][kk][kk+ee]=scene_graph_e[k][kk][kk+ee]=1
               risk_inter1=np.zeros((a.shape[1],a.shape[1]))
               risk_inter1[a.shape[1]-1,a.shape[1]-1]=0
               
               for i in range(a.shape[1]):
                   if i in ped_ii :# for pedestrians
                       if (sa_out[k][kk][-2] - sa_out[k][kk][-4] / 2) < a[k][i][0] < (
                               sa_out[k][kk][-2] - sa_out[k][kk][-4] / 2) or (
                               sa_out[k][kk][-1] - sa_out[k][kk][-3] / 2) < a[k][i][1] < (
                               sa_out[k][kk][-1] - sa_out[k][kk][-3] / 2):
                           if sa_out[k][kk][0] == 1.0 or sa_out[k][kk][1] == 1.0 or sa_out[k][kk][4] == 1.0:
                               for j in range(a.shape[1]):
                                   if i == j:
                                       risk_inter1[i, i] = 0
                                   else:
                                       d1 = np.sqrt((a[k][i][0] - a[k - 1][i][0]) * (a[k][i][0] - a[k - 1][i][0]) + (
                                                   a[k][i][1] - a[k - 1][i][1]) * (a[k][i][1] - a[k - 1][i][1]))
                                       v1 = d1 / 0.5
                                       d2 = np.sqrt((a[k][j][0] - a[k - 1][j][0]) * (a[k][j][0] - a[k - 1][j][0]) + (
                                                   a[k][j][1] - a[k - 1][j][1]) * (a[k][j][1] - a[k - 1][j][1]))
                                       v2 = d2 / 0.5
                                       angle1 = angle_l([a[k - 1][i], a[k][i]])
                                       angle2 = angle_l([a[k - 1][j], a[k][j]])
                                       dis = np.sqrt((a[k][j][0] - a[k][i][0]) * (a[k][j][0] - a[k][i][0]) + (
                                                   a[k][j][1] - a[k][i][1]) * (a[k][j][1] - a[k][i][1]))
                                       angle3 = angle_l([a[k][i], a[k][j]])
                                       if (angle1 - np.pi / 2) < angle3 < (angle1 + np.pi / 2):
                                           lij = 1
                                       else:
                                           lij = 0
                                       vv = abs(v1 * math.cos(abs(angle1 - angle3)) - v2 * math.cos(abs(angle2 - angle3)))
                                       t = dis / vv
                                       risk1 = 1 / t
#                                       bb = self.mlp(torch.Tensor([node_ou[k][i], node_ou[k][j]]))
                                      
                                       bb = self.mlp(torch.Tensor([node_ou[k][i], node_ou[k][j],clu[i],clu[j]]))
#                                       risk=bb*risk1
                                       risk = risk1 * bb * lij
                                       risk_inter1[i, j] = risk
                       else: 
                           for j in range(len(ped_ii)):
                               if i == j:
                                   risk_inter1[i, i] = 0 # OSR
                               else:
                                   d1 = np.sqrt((a[k][i][0] - a[k - 1][i][0]) * (a[k][i][0] - a[k - 1][i][0]) + (
                                           a[k][i][1] - a[k - 1][i][1]) * (a[k][i][1] - a[k - 1][i][1]))
                                   v1 = d1 / 0.5
                                   d2 = np.sqrt((a[k][j][0] - a[k - 1][j][0]) * (a[k][j][0] - a[k - 1][j][0]) + (
                                           a[k][j][1] - a[k - 1][j][1]) * (a[k][j][1] - a[k - 1][j][1]))
                                   v2 = d2 / 0.5
                                   angle1 = angle_l([a[k - 1][i], a[k][i]])
                                   angle2 = angle_l([a[k - 1][j], a[k][j]])
                                   dis = np.sqrt((a[k][j][0] - a[k][i][0]) * (a[k][j][0] - a[k][i][0]) + (
                                           a[k][j][1] - a[k][i][1]) * (a[k][j][1] - a[k][i][1]))
                                   angle3 = angle_l([a[k][i], a[k][j]])
                                   if (angle1 - np.pi / 2) < angle3 < (angle1 + np.pi / 2):
                                       lij = 1
                                   else:
                                       lij = 0
                                   vv = abs(v1 * math.cos(abs(angle1 - angle3)) - v2 * math.cos(abs(angle2 - angle3)))
                                   t = dis / vv
                                   risk1 = 1 / t
#                                   bb = self.mlp(torch.Tensor([node_ou[k][i], node_ou[k][j]]))
            
                                   bb = self.mlp(torch.Tensor([node_ou[k][i], node_ou[k][j],clu[i],clu[j]]))
#                                   risk=bb*risk1
                                   risk = risk1 * bb * lij  # risk computation between agents
                                   risk_inter1[i, j] = risk

                   elif i in veh_ii:  # for vehicles
                       if (sa_out[k][kk][-2] - sa_out[k][kk][-4] / 2) < a[k][i][0] < (
                               sa_out[k][kk][-2] - sa_out[k][kk][-4] / 2) or (
                               sa_out[k][kk][-1] - sa_out[k][kk][-3] / 2) < a[k][i][1] < (
                               sa_out[k][kk][-1] - sa_out[k][kk][-3] / 2):
                           if sa_out[k][kk][4] == 1.0 or sa_out[k][kk][5] == 1.0 or sa_out[k][kk][6] == 1.0:
                               for j in range(a.shape[1]):
                                   if i == j:
                                       risk_inter1[i, i] = 0 # OSR
                                   else:
                                       d1 = np.sqrt((a[k][i][0] - a[k - 1][i][0]) * (a[k][i][0] - a[k - 1][i][0]) + (
                                               a[k][i][1] - a[k - 1][i][1]) * (a[k][i][1] - a[k - 1][i][1]))
                                       v1 = d1 / 0.5
                                       d2 = np.sqrt((a[k][j][0] - a[k - 1][j][0]) * (a[k][j][0] - a[k - 1][j][0]) + (
                                               a[k][j][1] - a[k - 1][j][1]) * (a[k][j][1] - a[k - 1][j][1]))
                                       v2 = d2 / 0.5
                                       angle1 = angle_l([a[k - 1][i], a[k][i]])
                                       angle2 = angle_l([a[k - 1][j], a[k][j]])
                                       dis = np.sqrt((a[k][j][0] - a[k][i][0]) * (a[k][j][0] - a[k][i][0]) + (
                                               a[k][j][1] - a[k][i][1]) * (a[k][j][1] - a[k][i][1]))
                                       angle3 = angle_l([a[k][i], a[k][j]])
                                       if (angle1 - np.pi / 2) < angle3 < (angle1 + np.pi / 2):
                                           lij = 1
                                       else:
                                           lij = 0
                                       vv = abs(v1 * math.cos(abs(angle1 - angle3)) - v2 * math.cos(abs(angle2 - angle3)))
                                       t = dis / vv
                                       risk1 = 1 / t
#                                       bb = self.mlp(torch.Tensor([node_ou[k][i], node_ou[k][j]]))
                                      
                                       bb = self.mlp(torch.Tensor([node_ou[k][i], node_ou[k][j],clu[i],clu[j]]))
#                                       risk=bb*risk1
                                       risk = risk1 * bb * lij # risk computation between agents
                                       risk_inter1[i, j] = risk
                       else:
                           for j in range(len(veh_ii)):
                               if i == j:
                                   risk_inter1[i, i] = 0 # OSR
                               else:
                                   d1 = np.sqrt((a[k][i][0] - a[k - 1][i][0]) * (a[k][i][0] - a[k - 1][i][0]) + (
                                           a[k][i][1] - a[k - 1][i][1]) * (a[k][i][1] - a[k - 1][i][1]))
                                   v1 = d1 / 0.5
                                   d2 = np.sqrt((a[k][j][0] - a[k - 1][j][0]) * (a[k][j][0] - a[k - 1][j][0]) + (
                                           a[k][j][1] - a[k - 1][j][1]) * (a[k][j][1] - a[k - 1][j][1]))
                                   v2 = d2 / 0.5
                                   angle1 = angle_l([a[k - 1][i], a[k][i]])
                                   angle2 = angle_l([a[k - 1][j], a[k][j]])
                                   dis = np.sqrt((a[k][j][0] - a[k][i][0]) * (a[k][j][0] - a[k][i][0]) + (
                                           a[k][j][1] - a[k][i][1]) * (a[k][j][1] - a[k][i][1]))
                                   angle3 = angle_l([a[k][i], a[k][j]])
                                   if (angle1 - np.pi / 2) < angle3 < (angle1 + np.pi / 2):
                                       lij = 1
                                   else:
                                       lij = 0
                                   vv = abs(v1 * math.cos(abs(angle1 - angle3)) - v2 * math.cos(abs(angle2 - angle3)))
                                   t = dis / vv
                                   risk1 = 1 / t #TTC
#                                   bb = self.mlp(torch.Tensor([node_ou[k][i], node_ou[k][j]]))
                                   
                                   bb = self.mlp(torch.Tensor([node_ou[k][i], node_ou[k][j],clu[i],clu[j]]))
#                                   risk=bb*risk1
                                   risk = risk1 * bb * lij # risk computation between agents
                                   risk_inter1[i, j] = risk

           risk_inter.append(risk_inter1)
       
        risk_inter_out1=[risk_inter[0]]
        for jj in risk_inter:
            risk_inter_out1.append(jj)
        risk_inter2=np.array(risk_inter_out1)
        risk_inter_out=torch.from_numpy(risk_inter2).type(torch.float)
    
        return risk_inter_out,scene_graph_a,scene_graph_e,node_ou_64

class ConvTemporalGraphical(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super(ConvTemporalGraphical,self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A): 
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc//self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous(), A
    
class st_gcn(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 use_mdn = False,
                 stride=1,
                 dropout=0,
                 residual=True):
        super(st_gcn,self).__init__()
        

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])# gcn, tcn
        

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )
        self.prelu = nn.PReLU()
    def forward(self, x, A): #stgcnn
        res = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x)+res 
        if not self.use_mdn:
            x = self.prelu(x)
        return x, A

class social_stgcnn(nn.Module):
    def __init__(self,n_stgcnn =1,n_txpcnn=5,input_feat=2,output_feat=5,input_feat_seg=12,output_feat_seg=5,
                 seq_len=8,pred_seq_len=12,kernel_size=3):
        super(social_stgcnn,self).__init__()
        self.n_stgcnn= n_stgcnn
        self.n_txpcnn = n_txpcnn
        
        self.st_gcns = nn.ModuleList()
        self.st_gcns.append(st_gcn(input_feat,output_feat,(kernel_size,seq_len)))
        for j in range(1, self.n_stgcnn):
            self.st_gcns.append(st_gcn(output_feat,output_feat,(kernel_size,seq_len)))
                        
        
        self.st_gcns_seg = nn.ModuleList()
        self.st_gcns_seg.append(st_gcn(input_feat_seg,output_feat_seg,(kernel_size,seq_len)))
        for j in range(1, self.n_stgcnn):
            self.st_gcns_seg.append(st_gcn(output_feat_seg,output_feat_seg,(kernel_size,seq_len)))
                        
        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Conv2d(seq_len,pred_seq_len,3,padding=1))
        for j in range(1,self.n_txpcnn):
            self.tpcnns.append(nn.Conv2d(pred_seq_len,pred_seq_len,3,padding=1))
        self.tpcnn_ouput = nn.Conv2d(pred_seq_len,pred_seq_len,3,padding=1)
        
        self.prelus = nn.ModuleList()
        for j in range(self.n_txpcnn):
            self.prelus.append(nn.PReLU())
        self.risk_interaction=risk_interaction()
            
    def forward(self,cluster,obs_traj,obs_traj_rel,pred_traj_gt,start,pred_traj_gt_rel,end,sa_out,se_out,pedestrian_index,vehicle_index,rider_index):
    
        risk_out,sg_a,sg_e,node_ou_64=self.risk_interaction(cluster,obs_traj[start:end,:],start,end,sa_out,se_out,pedestrian_index,vehicle_index,rider_index)
        
        node_ou_64=node_ou_64.unsqueeze(0)
        node_ou_64=node_ou_64.permute(0,3,1,2)

        norm_lap_matr=True
        v_obs,a_ = seq_to_graph(obs_traj[start:end,:],obs_traj_rel[start:end, :],norm_lap_matr)
        V_obs=v_obs.unsqueeze(0)
        V_obs_tmp =V_obs.permute(0,3,1,2)
        Sg_a=sg_a.unsqueeze(0)
        Sg_a_tmp =Sg_a.permute(0,3,1,2)


        for k in range(self.n_stgcnn):
            v1,a = self.st_gcns[k](node_ou_64,risk_out)
#            v1,a = self.st_gcns[k](V_obs_tmp,risk_out)
        for k in range(self.n_stgcnn):
            v2,seg=self.st_gcns_seg[k](Sg_a_tmp,sg_e)

        v1 = v1.view(v1.shape[0],v1.shape[2],v1.shape[1],v1.shape[3])  #torch.Size([1, 4, 5, 3])# graph embedding of HRG
        v2 = v2.view(v2.shape[0],v2.shape[2],v2.shape[1],v2.shape[3])#torch.Size([1, 4, 5, 20])# graph embedding of HSG
        #v2_emp = v2[:,:,:,:8]
        conv_zc = nn.Conv2d(v2.shape[3],## convert the dimension of HSG embedding to be with the same dimension to HRG
             v1.shape[3],
             kernel_size=1,
             stride=(1,1),
             padding=0,
             dilation=1,
             bias=True)
        v2=v2.permute(0,3,2,1)
        y=conv_zc(v2)
        y=y.permute(0,3,2,1)

        v=v1
        v = self.prelus[0](self.tpcnns[0](v))
        for k in range(1,self.n_txpcnn-1):
            v =  self.prelus[k](self.tpcnns[k](v)) + v
        v = self.tpcnn_ouput(v)
        v = v.view(v.shape[0],v.shape[2],v.shape[1],v.shape[3])
        return v,a,risk_out









