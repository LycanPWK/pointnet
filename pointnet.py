
import torch as t
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import torch.utils.data as dat
from torch.autograd import Variable
import torch.optim as optim
batsz=5
epochs=1
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
        super(t_net_2,self).__init__()
        self.fc3=(256,4096)
    def forward(self, x):
        return super().forward(x)

class pointnet_cla(t_net_1):
    def __init__(self):
        
        super(pointnet_cla,self).__init__()
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
class pointnet_tr(pointnet_cla):
    def __init__(self):
        
        super(pointnet_tr,self).__init__()
        nn.dropout(p=0.3)
    
    def forward(self, x):
        x=super().forward(x)
        return f.log_softmax(x,dim=1)

def dataset_prepare():
    points=np.load("data.npy")
    label=np.load("label.npy")
    train_dataset=dat.TensorDataset(points,label)
    train_loader = dat.DataLoader(dataset=train_dataset, 
                                           batch_size=batsz, 
                                           shuffle=True)
    return train_loader
def rightness(predictions, labels):
    """计算预测错误率的函数，其中predictions是模型给出的一组预测结果，batch_size行num_classes列的矩阵，labels是数据之中的正确答案"""
    pred = t.max(predictions.data, 1)[1] # 对于任意一行（一个样本）的输出值的第1个维度，求最大，得到每一行的最大元素的下标
    rights = pred.eq(labels.data.view_as(pred)).sum() #将下标与labels中包含的类别进行比较，并累计得到比较正确的数量
    return rights, len(labels) #返回正确的数量和这一次一共比较了多少元素
def main():
    train_loader=dataset_prepare()
    net = pointnet_tr()
    criterion = nn.CrossEntropyLoss() #Loss函数的定义，交叉熵
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) #定义优化器，普通的随机梯度下降算法

    record = [] #记录准确率等数值的容器
    weights = [] #每若干步就记录一次卷积核

    #开始训练循环

    for epoch in range(epochs):
        print(epoch)
        train_rights = [] #记录训练数据集准确率的容器
        
        for batch_idx, (data, target) in enumerate(train_loader):  #针对容器中的每一个批进行循环
            net.train() 
            output = net(data) #神经网络完成一次前馈的计算过程，得到预测输出output
            loss = criterion(output, target) #将output与标签target比较，计算误差
            optimizer.zero_grad() #清空梯度
            loss.backward() #反向传播
            optimizer.step() #一步随机梯度下降算法
            right = rightness(output, target) #计算准确率所需数值，返回数值为（正确样例数，总样本数）
            train_rights.append(right) #将计算结果装到列表容器train_rights中

        
            if batch_idx % 100 == 0: #每间隔100个batch执行一次打印等操作
                
                net.eval() # 模型在测试集上
                val_rights = [] #记录校验数据集准确率的容器
                
                '''开始在校验数据集上做循环，计算校验集上面的准确度'''
                for (data, target) in train_loader:
                    output = net(data) #完成一次前馈计算过程，得到目前训练得到的模型net在校验数据集上的表现
                    right = rightness(output, target) #计算准确率所需数值，返回正确的数值为（正确样例数，总样本数）
                    val_rights.append(right)
                
                train_r = (sum([tup[0] for tup in train_rights]), sum([tup[1] for tup in train_rights]))
                test_r = (sum([tup[0] for tup in val_rights]), sum([tup[1] for tup in val_rights]))
                
    #             #打印准确率等数值，其中正确率为本训练周期Epoch开始后到目前撮的正确率的平均值
                print('训练周期: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t训练正确率: {:.2f}%\t校验正确率: {:.2f}%'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(), 
                    100. * train_r[0] / train_r[1], 
                    100. * test_r[0] / test_r[1]))
                
                #将准确率和权重等数值加载到容器中，以方便后续处理
                record.append((100 - 100. * train_r[0] / train_r[1], 100 - 100. * test_r[0] / test_r[1]))

                
