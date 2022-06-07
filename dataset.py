import open3d as o3d
import os
import igl
import train as tt
import numpy as np
path="dataset/"
label=[]
points=[]
cll=os.listdir(path)

for cl in cll:
    fll=os.listdir(path+"/"+cl+"/"+"test/")
    for fl in fll:
        if(".off" not in fl):
            continue
        mesh = o3d.io.read_triangle_mesh(path+"/"+cl+"/"+"test/"+fl)
        pcd=o3d.geometry.PointCloud()
        V_mesh = np.asarray(mesh.vertices)
        pcd = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, number_of_points=tt.sz)
        pc = np.asarray(pcd.points)
        label.append(cll.index(cl))
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        points.append(pc)
np.save("data1.npy",np.asarray(points))
np.save("label1.npy",np.asarray(label))
for cl in cll:
    fll=os.listdir(path+"/"+cl+"/"+"train/")
    for fl in fll:
        if(".off" not in fl):
            continue
        mesh = o3d.io.read_triangle_mesh(path+"/"+cl+"/"+"train/"+fl)
        pcd=o3d.geometry.PointCloud()
        pcd = o3d.geometry.TriangleMesh.sample_points_uniformly(mesh, number_of_points=tt.sz)
        pc = np.asarray(pcd.points)
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        points.append(pc)
        label.append(cll.index(cl))
np.save("data.npy",np.asarray(points))
np.save("label.npy",np.asarray(label))