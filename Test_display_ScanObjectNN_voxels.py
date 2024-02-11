import numpy as np
import os
import h5py
import glob
import open3d as o3d
import multiprocessing
from tqdm import tqdm
DATA_PATH = 'object_dataset/'

SAVE_PATH = 'processed_object_dataset/'

def data_generator(file_path, batch_size):
     with h5py.File(file_path, 'r') as f:
        total_size = f['data'].shape[0]
        for start_idx in range (0,total_size, batch_size):
            end_idx = min(start_idx + batch_size, total_size)
            yield f['data'][start_idx: end_idx], f['label'][start_idx: end_idx]


generator  = data_generator(SAVE_PATH+"voxel.h5",10)
voxel_arrays, labels = next(generator)
new_pcd = o3d.geometry.PointCloud()

voxel_array=voxel_arrays[0]
voxel_size = 224
# 遍历体素数组
points = []
colors = []
#for (224,224,224,3)
for x in range(voxel_array.shape[0]):
    for y in range(voxel_array.shape[1]):
        for z in range(voxel_array.shape[2]):
            color = voxel_array[x, y, z]
            if np.any(color > 0):  # 检查颜色是否不是全黑
                points.append([x, y, z])
                colors.append(color)
print(labels[0])
# 将点和颜色添加到点云对象
new_pcd.points = o3d.utility.Vector3dVector(np.array(points) * voxel_size)  # 缩放点到原始大小
new_pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
o3d.visualization.draw_geometries([new_pcd])