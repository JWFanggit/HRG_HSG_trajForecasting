# Heterogeneous Trajectory Forecasting via Risk and Scene Graph Learning
Heterogeneous trajectory forecasting is critical for intelligent transportation systems, but it is challenging because of the difficulty of modeling the complex interaction relations among the heterogeneous road agents as well as their agent-environment constraints.  In this work, we propose a risk and scene graph learning method for trajectory forecasting of heterogeneous road agents, which consists of a Heterogeneous Risk Graph (HRG) and a Hierarchical Scene Graph (HSG) from the aspects of agent category and their movable semantic regions. HRG groups each kind of road agent and calculates their interaction adjacency matrix based on an effective collision risk metric. HSG of the driving scene is modeled by inferring the relationship between road agents and road semantic layout aligned by the \emph{road scene grammar}. Based on this formulation, we can obtain effective trajectory forecasting in driving situations. 

![image](https://github.com/JWFanggit/HRG_HSG_trajForecasting/blob/main/RISG.jpg)

Because of the space limitation, we only provide the model code for nuScenes dataset training and testing. The results, datasets and other dataset evaluation code can be downloaded [Here](https://pan.baidu.com/s/12wq34ur-YvgIp3r0F3FD4A?pwd=39hl)(Extraction code:39hl)
