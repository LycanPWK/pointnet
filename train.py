import torch as t
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import torch.utils.data as dat
from torch.autograd import Variable
import torch.optim as optim
batsz=50
epochs=20
sz=512
class_num=10
class t_net_1(nn.Module):
    def __init__(self):
        super(t_net_1,self).__init__()

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
        self.maxpool=nn.MaxPool1d(sz)
    def forward(self,x):
        
        x=x.transpose(2,1)
        x=f.relu(self.bn3(self.mlpl1(x)))
        x=f.relu(self.bn2(self.mlpl2(x)))
        x=f.relu(self.bn1(self.mlpl3(x)))
   
        x=self.maxpool(x)
        x=x.reshape(-1,1024)
        x=f.relu(self.bn4(self.fc1(x)))
        x=f.relu(self.bn5(self.fc2(x)))
        x=self.fc3(x)

        iden = Variable(t.from_numpy(np.eye(3).flatten().astype(np.float32))).view(9).repeat(x.size()[0],1).cuda()
        x=x+iden
        return x

class t_net_2(nn.Module):
    def __init__(self):
        super(t_net_2,self).__init__()
        self.fc1=nn.Linear(1024,512)
        self.fc2=nn.Linear(512,256)
        self.fc3=nn.Linear(256,4096)

        self.bn1=nn.BatchNorm1d(1024)
        self.bn2=nn.BatchNorm1d(128)
        self.bn3=nn.BatchNorm1d(64)
        self.bn4=nn.BatchNorm1d(512)
        self.bn5=nn.BatchNorm1d(256)
        self.maxpool=nn.MaxPool1d(sz)
        self.mlpl1=nn.Conv1d(3,64,1)
        self.mlpl2=nn.Conv1d(64,128,1)
        self.mlpl3=nn.Conv1d(128,1024,1)
    def forward(self, x):
        x=x.transpose(2,1)
        x=f.relu(self.bn2(self.mlpl2(x)))
        x=f.relu(self.bn1(self.mlpl3(x)))
        x=self.maxpool(x)
        x=x.reshape(-1,1024)
        x=f.relu(self.bn4(self.fc1(x)))
        x=f.relu(self.bn5(self.fc2(x)))
        x=self.fc3(x)
        iden = Variable(t.from_numpy(np.eye(64).flatten().astype(np.float32))).view(4096).repeat(x.size()[0],1).cuda()
        x=x+iden
        return x
        
class pointnet_cla(nn.Module):
    def __init__(self):
        super(pointnet_cla,self).__init__()
        self.fc1=nn.Linear(1024,512)
        self.fc2=nn.Linear(512,256)

        self.bn1=nn.BatchNorm1d(1024)
        self.bn2=nn.BatchNorm1d(128)
        self.bn3=nn.BatchNorm1d(64)
        self.bn4=nn.BatchNorm1d(512)
        self.bn5=nn.BatchNorm1d(256)
        self.maxpool=nn.MaxPool1d(sz)
        self.mlpl1=nn.Conv1d(3,64,1)
        self.mlpl2=nn.Conv1d(64,128,1)
        self.mlpl3=nn.Conv1d(128,1024,1)
        
        self.t1net=t_net_1()
        self.t2net=t_net_2()
        self.fc3=nn.Linear(256,class_num)        
    def forward(self, x):
        x=t.bmm(x,(self.t1net(x).view(-1,3,3)))
        x=f.relu(self.bn3(self.mlpl1(x.transpose(1,2)))).transpose(2,1)
        x=t.bmm(x,(self.t2net(x).view(-1,64,64)))
        x=f.relu(self.bn2(self.mlpl2(x.transpose(1,2)))).transpose(2,1)
        x=f.relu(self.bn1(self.mlpl3(x.transpose(1,2))))
        x=self.t1net.maxpool(x).view(-1,1024)
        x=f.relu(self.bn4(self.fc1(x)))
        x=f.relu(self.bn5(self.fc2(x)))
        x=self.fc3(x)
        self.dropout=nn.Dropout(p=0.3)
        x=f.log_softmax(x)
        return x
def rightness(predictions, labels):
    pred = t.max(predictions.data, 1)[1] 
    rights = pred.eq(labels.data.view_as(pred)).sum() 
    return rights, len(labels) 
def main():

    points=np.load("data.npy")
    label=np.load("label.npy")
    points1=np.load("data1.npy")
    label1=np.load("label1.npy")
    train_dataset=dat.TensorDataset(t.tensor(points).to(t.float).cuda(),t.tensor(label).to(t.int64).cuda())
    test_dataset=dat.TensorDataset(t.tensor(points1).to(t.float).cuda(),t.tensor(label1).to(t.int64).cuda())
    train_loader = dat.DataLoader(dataset=train_dataset, 
                                           batch_size=batsz, 
                                           shuffle=True)
    test_loader=dat.DataLoader(dataset=test_dataset, 
                                           batch_size=batsz, 
                                           shuffle=True)                                      
    net = pointnet_cla()
    net.cuda()
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999)) 
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    record = [] 
    weights = [] 


    for epoch in range(epochs):
        print(epoch)
        train_rights = [] 
        scheduler.step()
        for batch_idx, (data, target) in enumerate(train_loader):  
            net=net.train()
            output = net(data) 
            loss = criterion(output,target) 
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 
            right = rightness(output,target) 
            train_rights.append(right) 

        
            if batch_idx % 10 == 0: 
                
                net=net.eval() 
                val_rights = [] 
                

                for (data, target) in test_loader:
                    output = net(data) 
                    right = rightness(output, target) 
                    val_rights.append(right)
                
                train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
                test_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))
                
                print('epochs: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\ttest AC: {:.2f}%\tval AC: {:.2f}%'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(), 
                    100. * train_r[0] / train_r[1], 
                    100. * test_r[0] / test_r[1]))
                
       
                record.append((100 - 100. * train_r[0] / train_r[1], 100 - 100. * test_r[0] / test_r[1]))
                
if __name__=="__main__":
    main()
                
