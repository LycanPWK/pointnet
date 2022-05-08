import torch as t
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import dataset
batsz=5
epochs=1
class t_net_1(nn.modules):
    def __init__(self):
        super().__init__()

        self.fc1=nn.Linear(1024,512)
        self.fc2=nn.Linear(512,256)
        self.fc3=nn.Linear(256,9)

        self.bn1=nn.BatchNorm1d(1024)
        self.bn2=nn.BatchNorm1d(128)
        self.bn3=nn.BatchNorm1d(64)
        self.bn4=nn.BatchNorm1d(512)
        self.bn5=nn.BatchNorm1d(256)

        self.mlpl1=nn.Conv1d(3,64,1)
        self.mlpl2=nn.Conv1d(64,128,1)
        self.mlpl3=nn.Conv1d(128,1024,1)
    
    def forward(self,x):
        x=nn.ReLU(self.bn3(self.mlpl1(x)))
        x=nn.ReLU(self.bn2(self.mlpl2(x)))
        x=nn.ReLU(self.bn1(self.mlpl3(x)))
        x=nn.MaxPool1d(1024)(x)
        x=nn.ReLU(self.bn4(self.fc1(x)))
        x=nn.ReLU(self.bn5(self.fc2(x)))
        x=t.self.fc3(x)
        return x

class t_net_2(t_net_1):
    def __init__(self):
        super().__init__()
        self.fc3=(256,4096)
    def forward(self, x):
        return super().forward(x)

class pointnet_classification(t_net_1):
    def __init__(self):
        
        super().__init__()
        self.t1net=t_net_1()
        self.t2net=t_net_2()
        self.fc3=(256,10)
 
    def forward(self, x):

        x=t.bmm(x,(self.t1net(x).reshape(3,3)))
        x=nn.ReLU(self.bn3(self.mlpl1(x)))
        x=t.bmm(x,self.t2net(x).reshape(64,64))
        x=nn.ReLU(self.bn1(self.mlpl3(x)))
        x=nn.MaxPool1d(1024)(x)
        x=nn.ReLU(self.bn4(self.fc1(x)))
        x=nn.ReLU(self.bn5(self.fc2(x)))
        x=t.self.fc3(x)
        return x

def main():
    dataset_prepare()
    net=pointnet_classification()
