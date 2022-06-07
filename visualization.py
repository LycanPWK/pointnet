from copyreg import pickle
from operator import indexOf
import torch as t
import torch.nn as nn
import torch.nn.functional as f
import open3d as o3d
import numpy as np
import os
import pickle
import train as tt
import torch.utils.data as dat
from torch.autograd import Variable
import torch.optim as optim
def inderightness(predictions, labels):
    pred = t.max(predictions.data, 1)[1] 
    rights = pred.eq(labels.data.view_as(pred))
    return rights,pred
points1=np.load("data1.npy")
label1=np.load("label1.npy")
test_dataset=dat.TensorDataset(t.tensor(points1).to(t.float).cuda(),t.tensor(label1).to(t.int64).cuda())
test_loader=dat.DataLoader(dataset=test_dataset, 
                                        batch_size=1, 
                                        shuffle=False)   
net = tt.pointnet_cla()
net.load_state_dict(t.load('10.pt'))
net.cuda()
net=net.eval()
with open("meshes.txt","r") as f:
    meshes=f.read().split('\n')
i=0
for (tdata, target) in test_loader:
    output = net(tdata)
    rights=inderightness(output.cpu(),target.cpu())
    ind,pred=np.asarray(rights)
    p=np.asarray(pred.cpu())
    l=np.asarray(target.cpu())
    if(l!=p):
        print(p,l,'\n')
        pcd=o3d.open3d.geometry.PointCloud()
        mesh=o3d.io.read_triangle_mesh(meshes[i])
        pcd = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, number_of_points=tt.sz)
        pcd.paint_uniform_color([1.0, 0.5, 0.0])
        mesh.compute_vertex_normals()
        o3d.visualization.draw_geometries([pcd,mesh])
    i+=1
    tdata.cuda()
    target.cuda()