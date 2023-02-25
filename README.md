# Heterogeneous Trajectory Forecasting via Risk and Scene Graph Learning, submited to IEEE-TITS.
Heterogeneous trajectory forecasting is critical for intelligent transportation systems, but it is challenging because of the difficulty of modeling the complex interaction relations among the heterogeneous road agents as well as their agent-environment constraints.  In this work, we propose a risk and scene graph learning method for trajectory forecasting of heterogeneous road agents, which consists of a Heterogeneous Risk Graph (HRG) and a Hierarchical Scene Graph (HSG) from the aspects of agent category and their movable semantic regions. HRG groups each kind of road agent and calculates their interaction adjacency matrix based on an effective collision risk metric. HSG of the driving scene is modeled by inferring the relationship between road agents and road semantic layout aligned by the \emph{road scene grammar}. Based on this formulation, we can obtain effective trajectory forecasting in driving situations. 

![image](https://github.com/JWFanggit/HRG_HSG_trajForecasting/blob/main/RISG.jpg)

Because of the space limitation, we only provide the model code for nuScenes dataset training and testing. The results, datasets and other dataset evaluation code can be downloaded [Here](https://pan.baidu.com/s/12wq34ur-YvgIp3r0F3FD4A?pwd=39hl)(Extraction code:39hl)

### RISG

We propose an interaction graph construction method based on trajectory clustering is proposed to extract the interaction characteristics between pedestrians and a risk and scene graph (RISG) learning method for heterogeneous road agent trajectory prediction is proposed. Experiments were performed on the nuScenes dataset, Apolloscape dataset, and Argoverse dataset, and the results showed a 48\% improvement in FDE and a 28\% improvement in ADE compared to S-STGCNN. It is proved that this method can improve the trajectory prediction accuracy of heterogeneous traffic participants.

#### Model

RISG model consists of 2 building blocks: <br />
1- HSG:  An interaction graph construction method based on trajectory clustering to extract the interaction features between pedestrians.<br />
2- RISG:  Presents a RIsk and Scene Graph (RISG) learning method for trajectory prediction of heterogeneous road agents.<br />

### Setup: 
The code was written using python 3.8. 
The following libraries are the minimal to run the code: 
```python
import pytorch
import networkx
import numpy
import tqdm
```
or you can have everything set up by running: 
```bash
pip install -r requirements.txt
```

### Using the code:

To train a model for each data set with the best configuration as in the paper, simply run:
```bash
train.py 
```

To use the pretrained models at `checkpoint_zc_/` :
```bash
zc_test.py
```


Please note: The initiation of training and testing might take a whileas as the code will pre-train and test after all the data is loaded.
