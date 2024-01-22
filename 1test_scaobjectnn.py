import h5py
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import sys

# open the file for reading
# training_dataset_path = '/data/ScanObjectNN/h_5files/main_split/training_objectdataset.h5'
# with h5py.File(training_dataset_path, 'r') as f:
  
#     # list all datasets in the file
#     print(list(f.keys()))
  
#     # access a specific dataset
#     training_data = f['data']# shape: (2309,2048,3) 
#     training_label = f['label']# shape: (2309,)
#     training_mask = f['mask'] # shape: (2309,2048)
  
#     # read the data from the dataset
#     data = training_data[:]
#     label = training_label[:]
#     # do something with the data
#     print(label)

#sys.exit()


def visualize_voxels_with_open3d(voxel_array):
    # 创建坐标系网格

    new_pcd = o3d.geometry.PointCloud()
    voxel_size = 1 / 224  # 体素大小
    # 遍历体素数组
    points = []
    colors = []
    for x in range(voxel_array.shape[0]):
        for y in range(voxel_array.shape[1]):
            for z in range(voxel_array.shape[2]):
                color = voxel_array[x, y, z]
                if np.any(color > 0):  # 检查颜色是否不是全黑
                    points.append([x, y, z])
                    colors.append(color)

    # 将点和颜色添加到点云对象
    new_pcd.points = o3d.utility.Vector3dVector(np.array(points) * voxel_size)  # 缩放点到原始大小
    new_pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
    o3d.visualization.draw_geometries([new_pcd])
def visualize_voxels_with_matplot(data):
    # 选择要显示的切片（例如中间切片）
    slice_x = data[112, :, :, :]
    slice_y = data[:, 112, :, :]
    slice_z = data[:, :, 112, :]

    # 创建图形和子图
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # 显示每个切片
    axs[0].imshow(slice_x)
    axs[0].set_title('Slice on X axis')

    axs[1].imshow(slice_y)
    axs[1].set_title('Slice on Y axis')

    axs[2].imshow(slice_z)
    axs[2].set_title('Slice on Z axis')

    # 显示图形
    plt.show()
file = 'bed_voxel.h5'
with h5py.File(file, 'r') as f:
  
    # list all datasets in the file
    print(list(f.keys()))
  
    # access a specific dataset
    voxel_array = f['data']# shape: (2309,2048,3) 
    visualize_voxels_with_open3d(voxel_array)
    visualize_voxels_with_matplot(voxel_array)