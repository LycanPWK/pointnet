import torch as t
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import torch.utils.data as dat
from torch.autograd import Variable
import torch.optim as optim
from torch.autograd import Variable
batsz=200
epochs=10
sz=256
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
        self.t1net=t_net_1()
        self.t2net=t_net_2()
        self.fc3=nn.Linear(256,class_num)        
    def forward(self, x):
        x=t.bmm(x,(self.t1net(x).view(-1,3,3)))
        x=f.relu(self.t1net.bn3(self.t1net.mlpl1(x.transpose(1,2)))).transpose(2,1)
        x=t.bmm(x,(self.t2net(x).view(-1,64,64)))
        x=f.relu(self.t1net.bn2(self.t1net.mlpl2(x.transpose(1,2)))).transpose(2,1)
        x=f.relu(self.t1net.bn1(self.t1net.mlpl3(x.transpose(1,2))))
        x=self.t1net.maxpool(x).view(-1,1024)
        x=f.relu(self.t1net.bn4(self.t1net.fc1(x)))
        x=f.relu(self.t1net.bn5(self.t1net.fc2(x)))
        x=self.fc3(x)
        self.dropout=nn.Dropout(p=0.3)
        x=f.log_softmax(x)
        return x
def rightness(predictions, labels):
    pred = t.max(predictions.data, 1)[1] # 对于任意一行（一个样本）的输出值的第1个维度，求最大，得到每一行的最大元素的下标
    rights = pred.eq(labels.data.view_as(pred)).sum() #将下标与labels中包含的类别进行比较，并累计得到比较正确的数量
    return rights, len(labels) #返回正确的数量和这一次一共比较了多少元素
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
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)            

    for epoch in range(epochs):
        scheduler.step()
        for i, data in enumerate(train_loader, 0):
            points, target = data
            points, target = points.cuda(), target.cuda()
            optimizer.zero_grad()
            net = net.train()
            pred = net(points)
            loss = f.nll_loss(pred, target)
            loss.backward()
            optimizer.step()
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, batsz, loss.item(), correct.item() / float(batsz)))

            if i % 10 == 0:
                j, data = next(enumerate(test_loader, 0))
                points, target = data
                points, target = points.cuda(), target.cuda()
                net = net.eval()
                pred = net(points)
                loss = f.nll_loss(pred, target)
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                blue = lambda x: '\033[94m' + x + '\033[0m'
                print('[%d: %d/%d] %s loss: %f accuracy: %f' % (epoch, i, batsz, blue('test'), loss.item(), correct.item()/float(batsz)))

        #t.save(net.state_dict(), '%s/cls_model_%d.pth' % ('datasets/', epoch))

    total_correct = 0
    total_testset = 0
    for i,data in tqdm(enumerate(test_loader, 0)):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        classifier = classifier.eval()
        pred, _, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        total_correct += correct.item()
        total_testset += points.size()[0]

    print("final accuracy {}".format(total_correct / float(total_testset)))
                
if __name__=="__main__":
    main()
