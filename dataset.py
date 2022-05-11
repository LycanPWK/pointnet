import open3d as o3d
import os
import igl
import numpy as np
import torch.utils.data as dat
path="dataset/"
label=[]
points=[]
sampl_num=512
cll=os.listdir(path)
for cl in cll:
    fll=os.listdir(path+"/"+cl+"/"+"test/")
    for fl in fll:
        if(".off" not in fl):
            continue
        mesh = o3d.io.read_triangle_mesh(path+"/"+cl+"/"+"test/"+fl)
        pcd=o3d.geometry.PointCloud()
        V_mesh = np.asarray(mesh.vertices)
        pcd = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, number_of_points=sampl_num)
        points.append(np.asarray(pcd.points))
        label.append(cll.index(cl))
np.save("data1.npy",np.asarray(points))
np.save("label1.npy",np.asarray(label))
for cl in cll:
    fll=os.listdir(path+"/"+cl+"/"+"train/")
    for fl in fll:
        if(".off" not in fl):
            continue
        mesh = o3d.io.read_triangle_mesh(path+"/"+cl+"/"+"train/"+fl)
        pcd=o3d.geometry.PointCloud()
        V_mesh = np.asarray(mesh.vertices)
        pcd = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, number_of_points=sampl_num)
        points.append(np.asarray(pcd.points))
        label.append(cll.index(cl))
np.save("data.npy",np.asarray(points))
np.save("label.npy",np.asarray(label))