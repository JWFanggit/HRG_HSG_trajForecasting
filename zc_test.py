import os
import math
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import argparse
import glob
import torch.distributions.multivariate_normal as torchdist
from utils import * 
from zc_metrics import * 
from model import social_stgcnn,seq_to_graph
import copy
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
        
def test(KSTEPS=20):
    global batch_test,model
    model.eval()
    ade_bigls = []
    fde_bigls = []
    raw_data_dict = {}
    step =0 

    obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
    loss_mask,seq_start_end,sa,se,cluster,pedestrian_index,vehicle_index,rider_index= batch_test

    pedestrian_index=[i for i in pedestrian_index if i<10856 ]
    vehicle_index=[i for i in vehicle_index if i<10856 ]
    rider_index=[i for i in rider_index if i<10856 ]
    ant=0
    for ss in range(0,2000):
#    for ss in range(len(seq_start_end)):
        print('ss',ss)
        step+=1
        start, end = seq_start_end[ss]
        obs_len=obs_seq_len
        se_out=se[ss,0:obs_len,:]
        sa_out=sa[ss,0:obs_len,:]
        norm_lap_matr=True
        V_obs,a_ = seq_to_graph(obs_traj[start:end,:],obs_traj_rel[start:end, :],norm_lap_matr)

        v_tr,a_=seq_to_graph(pred_traj_gt[start:end,:],pred_traj_gt_rel[start:end, :],norm_lap_matr)
    

        V_pred,_,risk_out=model(cluster,obs_traj,obs_traj_rel,pred_traj_gt,start,pred_traj_gt_rel,end,sa_out,se_out,pedestrian_index,vehicle_index,rider_index)
        V_pred = V_pred.permute(0,2,3,1)
        V_pred = V_pred.squeeze()
        V_tr = v_tr
        num_of_objs = obs_traj_rel[start:end,:].shape[0]
        V_pred,V_tr =  V_pred[:,:num_of_objs,:],V_tr[:,:num_of_objs,:]
        
    
        sx = torch.exp(V_pred[:,:,2]) #sx
        sy = torch.exp(V_pred[:,:,3]) #sy
        corr = torch.tanh(V_pred[:,:,4]) #corr

        cov = torch.zeros(V_pred.shape[0],V_pred.shape[1],2,2)
        cov[:,:,0,0]= sx*sx
        cov[:,:,0,1]= corr*sx*sy
        cov[:,:,1,0]= corr*sx*sy
        cov[:,:,1,1]= sy*sy
        mean = V_pred[:,:,0:2]
        cov=abs(cov)

         
        b=torch.linalg.eigvals(cov)
        b=b.detach().numpy()
        
        if np.min(b)>0:
           cov=cov
        else:
            print('cuowu')
            index=[i for i in range(start,end)]
            for jjj in index:
                if jjj in pedestrian_index:
                    pedestrian_index.remove(jjj)
                elif jjj in vehicle_index:
                    vehicle_index.remove(jjj)
                else:
                    rider_index.remove(jjj)                   
            continue
        #Now sample 20 samples
        ade_ls = {}
        fde_ls = {}

        pred_ls={}
        pred_frame={}
        target_ls={}
        obsrvs_ls={}
        pred_ls_all={}
        V_x = seq_to_nodes(obs_traj[start:end,:].data.cpu().numpy().copy())
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(V_obs.data.cpu().numpy().squeeze().copy(),
                                                 V_x[0,:,:].copy())
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.data.cpu().numpy().squeeze().copy(),
                                                 V_x[-1,:,:].copy())
        raw_data_dict[step] = {}
        raw_data_dict[step]['obs'] = copy.deepcopy(V_x_rel_to_abs)
        raw_data_dict[step]['trgt'] = copy.deepcopy(V_y_rel_to_abs)
        raw_data_dict[step]['pred'] = []

        aaa=[]
        pred_ls_all=[]
        for j in range(mean.shape[0]): 
            if j==0:
                aa=V_x[-1,:,:].copy()
            for n in range(num_of_objs):
                ade_ls[n]=[]
                fde_ls[n]=[]
    
                pred_ls[n]=[]
                target_ls[n]=[]
                obsrvs_ls[n]=[]
                
            for k in range(KSTEPS):
                mvnormal = torchdist.MultivariateNormal(mean[j,:,:],cov[j,:,:,:])
                V_pred = mvnormal.sample()
                V_pred_rel_to_abs = nodes_rel_to_nodes_abs_frame(V_pred.data.cpu().numpy().squeeze().copy(),
                                                         aa)
                for n in range(num_of_objs):
                    pred = [] 
                    target = []
                    number_of = []
                    pred.append(V_pred_rel_to_abs[n:n+1,:])
                    target.append(V_y_rel_to_abs[j,n:n+1,:])
                    number_of.append(1)
                    
                    fde_frame1=fde_frame(pred,target,number_of)
                    ade_ls[n].append(fde_frame1)
                    pred_ls[n].append(pred)

            pred_min=[]       
            pred_ls_frame=[]
            for n in range(num_of_objs):
                q=ade_ls[n].index(min(ade_ls[n]))
                pred_min1=pred_ls[n][q]
                pred_min.append(pred_min1[0][0].tolist())
                pred_ls[n]=np.array(pred_ls[n]).squeeze()
                pred_ls_frame.append(pred_ls[n])
            aa=pred_min

            pred_ls[n]=np.array(pred_ls[n]).squeeze()
            
            pred_ls_all.append(pred_ls_frame)
            aaa.append(aa)

        aaa=torch.tensor(aaa)
        pred_ls_all=torch.tensor(pred_ls_all)
        
        
#        fig = plt.figure(figsize=(9, 6))
#        ax = fig.add_subplot(111)
        
        for n in range(num_of_objs):
            pred_ls[n]=np.array(pred_ls[n]).squeeze()

            pred = [] 
            target = []
            obsrvs = [] 
            number_of = []
            pred.append(aaa[:,n:n+1,:])
            target.append(V_y_rel_to_abs[:,n:n+1,:])
            obsrvs.append(V_x_rel_to_abs[:,n:n+1,:])
            number_of.append(1)

            ade1=ade(pred,target,number_of)
            ade_bigls.append(ade1)
            fde1=fde(pred,target,number_of)
            fde_bigls.append(fde1)
        
        
##########visualize the predicted trajectories
#            pred_min1=pred[0]
#            pred_min1_x=pred_min1[:,:,0].squeeze()
#            pred_min1_y=pred_min1[:,:,1].squeeze()
#            target_min1=target[0]
#            target_min1_x=target_min1[:,:,0].squeeze()
#            target_min1_y=target_min1[:,:,1].squeeze()
#            obsrvs_min1=obsrvs[0]
#            obsrvs_min1_x=obsrvs_min1[:,:,0].squeeze()
#            obsrvs_min1_y=obsrvs_min1[:,:,1].squeeze()
#
#            ax.plot(pred_min1_x, pred_min1_y, 'g--', lw=4.0, label="pred")  #画线
#            ax.plot(target_min1_x, target_min1_y, 'r--', lw=4.0, label="target")
#            ax.plot(obsrvs_min1_x, obsrvs_min1_y, 'b', lw=4.0, label="obsrvs")
#        name=str(ant)+'.png'
#        plt.savefig('./figure/'+'nuscenes/'+name)
#        ant+=1
###########save observation and groundtruth
#        print('V_x_rel_to_abs',V_x_rel_to_abs.shape)
#        print('V_y_rel_to_abs',V_y_rel_to_abs.shape)
#        ground_truth=np.concatenate([V_x_rel_to_abs,V_y_rel_to_abs],axis=0).squeeze()
#        ground_truth=torch.tensor(ground_truth).permute(1,0,2)
##        print('ground_truth',ground_truth.shape)  #[4, 10, 2]
#        ground_truth_zc=ground_truth.reshape(ground_truth.shape[0]*ground_truth.shape[1],ground_truth.shape[2])
##        print('ground_truth',ground_truth.shape)
#        ground_truth_zc1=pd.DataFrame(ground_truth_zc.detach().numpy())
#        name1=str(ss)+'.csv'
#        ground_truth_zc1.to_csv('./ground_truth_zc/'+name1)
###########save 20 times of prediction
#        pred_zc=torch.tensor(pred_ls_all).permute(1,0,2,3)
##        print('pred_zc1',pred_zc.shape)###[4, 6, 20, 2]
#        pred_zc=pred_zc.reshape(pred_zc.shape[0]*pred_zc.shape[1]*pred_zc.shape[2],pred_zc.shape[3])
#        pred_zc1=pd.DataFrame(pred_zc.detach().numpy())
#        name1=str(ss)+'.csv'
#        pred_zc1.to_csv('./pred_all_2/'+name1)
###########save predicted trajectories with minimum ADE
#        aaa=torch.tensor(aaa).permute(1,0,2)
#        pred_shuchu=aaa.reshape(aaa.shape[0]*aaa.shape[1],aaa.shape[2])
#        pred_shuchu=pd.DataFrame(pred_shuchu.detach().numpy())
#        name1=str(ss)+'.csv'
#        pred_shuchu.to_csv('./pred_min_2/'+name1)
#            
    zuizhong_index=pedestrian_index+vehicle_index+rider_index
    zuizhong_index=sorted(zuizhong_index)

    ade_p=[]
    fde_p=[]
    for j in pedestrian_index:
        qq=zuizhong_index.index(j)
        if fde_bigls[qq]<15:
            ade_p.append(ade_bigls[qq])
            fde_p.append(fde_bigls[qq])
    ade_ped = sum(ade_p)/len(ade_p)
    fde_ped = sum(fde_p)/len(ade_p)
    mr_ped = sum(np.array(fde_p) > 2 ) / len(fde_p)   #2 meters for miss rate
    print('len(ade_p)',len(ade_p))

    ade_v=[]
    fde_v=[]
    for j in vehicle_index:
        qq=zuizhong_index.index(j)
        if fde_bigls[qq]<20:
            ade_v.append(ade_bigls[qq])
            fde_v.append(fde_bigls[qq])
    ade_veh = sum(ade_v)/len(ade_v)
    fde_veh = sum(fde_v)/len(ade_v)
    mr_veh = sum(np.array(fde_v)>2) / len(fde_v)  #2 meters for miss rate
    print('len(ade_v)',len(ade_v))

    ade_r=[]
    fde_r=[]
    if len(rider_index)==0:
        ade_rider=0
        fde_rider=0
    else:
        for j in rider_index:
            qq=zuizhong_index.index(j)
            if fde_bigls[qq]<15:
                ade_r.append(ade_bigls[qq])
                fde_r.append(fde_bigls[qq])
        ade_rider = sum(ade_r)/len(ade_r)
        fde_rider = sum(fde_r)/len(ade_r)
        mr_rider = sum(np.array(fde_r)>2)/len(fde_r) #2 meters for miss rate
    print('len(ade_r)',len(ade_r))

    ade_ = (sum(ade_p)+sum(ade_v)+sum(ade_r))/(len(ade_p)+len(ade_v)+len(ade_r))
    fde_ = (sum(fde_p)+sum(fde_v)+sum(fde_r))/(len(fde_p)+len(fde_v)+len(fde_r))
    mr = (mr_rider*len(fde_r)+mr_veh*len(fde_v)+mr_ped*len(fde_p))/(len(ade_p)+len(ade_v)+len(ade_r))

    return ade_,fde_,mr,mr_ped,mr_veh,mr_rider,ade_ped,fde_ped,ade_veh,fde_veh,ade_rider,fde_rider,raw_data_dict

paths = ['./checkpoint_zc_res/tag']

KSTEPS=20

print("*"*50)
print('Number of samples:',KSTEPS)
print("*"*50)



for feta in range(len(paths)):
    ade_ls = [] 
    fde_ls = [] 

    path = paths[feta]
    exps = glob.glob(path)  
    print('Model being tested are:',exps)
    exp_path1=[]
    for exp_path in exps:
        exp_path1.append(exp_path)
        print("*"*50)
        print("Evaluating model:",exp_path)

        model_path = exp_path+'/val_best.pth'

        args_path = exp_path+'/args.pkl'
        with open(args_path,'rb') as f: 
            args = pickle.load(f)
        print('args.input_size',args.input_size)
        #Data prep     
        obs_seq_len = args.obs_seq_len
        pred_seq_len = args.pred_seq_len
        data_set = './datasets/eth'+'/'
        

        batch_test = TrajectoryDataset(
        data_set+'test/',         #train，load testing dataset
        obs_len=obs_seq_len,       
        pred_len=pred_seq_len,     
        skip=1,norm_lap_matr=True,
        type_='test')   #norm(v)    lap: Laplace  matrix 
        
        #Defining the model 
        model = social_stgcnn(n_stgcnn =args.n_stgcnn,n_txpcnn=args.n_txpcnn,
        input_feat=args.input_size,
        output_feat=args.output_size,
        input_feat_seg=args.input_size_seg,
        output_feat_seg=args.output_size_seg,
        seq_len=args.obs_seq_len,
        kernel_size=args.kernel_size,pred_seq_len=args.pred_seq_len)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        
        ade_ =999999
        fde_ =999999
        print("Testing ....")

        ad,fd,mr,mr_ped,mr_veh,mr_rider,ad_ped,fd_ped,ad_veh,fd_veh,ad_rider,fd_rider,raw_data_dict=test()
        ade_average= min(ade_,ad)
        fde_average =min(fde_,fd)
        ade_ped= min(ade_,ad_ped)
        fde_ped =min(fde_,fd_ped)
        ade_veh= min(ade_,ad_veh)
        fde_veh =min(fde_,fd_veh)
        ade_rider= min(ade_,ad_rider)
        fde_rider =min(fde_,fd_rider)

        print("ADE_average:",ade_average,'\n',
              " FDE_average:",fde_average,'\n',
              "ade_ped:",ade_ped,'\n',
              "fde_ped:",fde_ped,'\n',
              "ade_veh:",ade_veh,'\n',
              "fde_veh:",fde_veh,'\n',
              "ade_rider:",ade_rider,'\n',
              "fde_rider:",fde_rider,'\n',
              "mr:",mr,'\n',
              "mr_ped:",mr_ped,'\n',
              "mr_veh:",mr_veh,'\n',
              "mr_rider:",mr_rider,'\n')

        
        
        
        
        
        
