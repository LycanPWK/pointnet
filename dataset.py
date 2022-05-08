import open3d as o3d
import os
import igl
import numpy as np
import torch.utils.data as dat

path="../dataset/"
cll=os.listdir(path)
for cl in cll:
    fll=os.listdir(path+"/"+cl)
    for fl in fll:
        mesh = o3d.io.read_triangle_mesh(path+"/"+"train/"+cl"/"+fl)
        pcd=o3d.geometry.PointCloud()
        V_mesh = np.asarray(mesh.vertices)
        pcd = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, number_of_points=2048)
        points = np.concatenate(points,np.asarray(pcd.points),axis=2)
        label= np.concatenate(label,np.asarray(cl))
