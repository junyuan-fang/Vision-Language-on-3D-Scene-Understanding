import open3d as o3d
import numpy as np

# 假设 xyz 和 rgb 是你的数据，格式为 N x 3
xyz = np.random.rand(100, 3)  # 随机生成一些点的位置数据
rgb = np.random.rand(100, 3)  # 随机生成一些点的颜色数据

# 创建一个点云对象
pcd = o3d.geometry.PointCloud()

# 将 XYZ 和 RGB 数据设置到点云对象中
pcd.points = o3d.utility.Vector3dVector(xyz)
pcd.colors = o3d.utility.Vector3dVector(rgb)

# 下面你可以对点云进行处理，例如进行体素化（voxelization）
voxel_size = 0.05  # 设定体素的大小
voxel_grid = pcd.voxel_down_sample(voxel_size)
print(voxel_grid)