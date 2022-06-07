import open3d as o3d
import numpy as np
import os
import dill
path="dataset/"
cll=os.listdir(path)
meshes=[]
for cl in cll:
    fll=os.listdir(path+"/"+cl+"/"+"train/")
    for fl in fll:
        if(".off" not in fl):
            continue
        meshes.append(path+cl+"/"+"train/"+fl)
with open("meshes.txt","w") as f:
    f.write('\n'.join(meshes))        