import os
from clip import clip
import torch
from torchvision.datasets import STL10
import numpy as np
torch.cuda.empty_cache()
topk = 5

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

# Download the dataset
stl10_dataset = STL10(root=os.path.expanduser("./data"), download=False, split='test')
# Prepare the prompts
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in stl10_dataset.classes]).to(device)


image, class_id = stl10_dataset[2]
voxel_input = preprocess(image)#.unsqueeze(0).to(device)
voxel_input = voxel_input.permute(1,2,3,0) 
print(voxel_input.min(), voxel_input.max())

# import sys
# torch.cuda.empty_cache()
# sys.exit()

import open3d as o3d

pcd = o3d.geometry.PointCloud()

points = np.zeros((224*224*224,3))  # 提取坐标并重塑
colors = np.zeros((224*224*224,3))  # 提取坐标并重塑
index = 0
for i in range(0,224):
    
    for j in range(0,224):
        
        for k in range(0,224):
            #if voxel_input...
            points[index][0] = i
            points[index][1] = j
            points[index][2] = k
            colors[index][0] = voxel_input[i][j][k][0]
            colors[index][1] = voxel_input[i][j][k][1]
            colors[index][2] = voxel_input[i][j][k][2]
            index+=1
# 将你的数据转换为点云格式

pcd.points = o3d.utility.Vector3dVector(points)


pcd.colors = o3d.utility.Vector3dVector(colors)  # 归一化颜色值到 [0, 1]

# 可视化点云
o3d.visualization.draw_geometries([pcd])
torch.cuda.empty_cache()