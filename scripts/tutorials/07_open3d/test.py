import open3d as o3d
import open3d.core as o3c
import numpy as np

# Load Point Cloud Data
ply_point_cloud = o3d.data.PLYPointCloud()
pcd = o3d.io.read_point_cloud("C:/Users/grape4314/open3d_data/download/PLYPointCloud/fragment.ply")

# Voxelization Down Sampling
downpcd = pcd.voxel_down_sample(voxel_size=0.02)
# o3d.visualization.draw_geometries([downpcd.to_legacy()],
#                                   zoom=0.3412,
#                                   front=[0.4257, -0.2125, -0.8795],
#                                   lookat=[2.6172, 2.0475, 1.532],
#                                   up=[-0.0694, -0.9768, 0.2024], width=960, height=480)

# Farthest Point Samping (FPS)
downpcd_fps = downpcd.farthest_point_down_sample(num_samples=100)
# o3d.visualization.draw_geometries([downpcd_fps],
#                                   zoom=0.3412,
#                                   front=[0.4257, -0.2125, -0.8795],
#                                   lookat=[2.6172, 2.0475, 1.532],
#                                   up=[-0.0694, -0.9768, 0.2024], width=960, height=480)

# Generate KD Tree
pcd_tree = o3d.geometry.KDTreeFlann(pcd)

# KNN Search
patches = list()
print("Find it 50 nearest neighbors")
for i in range(100):      
    [k, idx, _] = pcd_tree.search_knn_vector_3d(downpcd_fps.points[i], 50)
    points = pcd.select_by_index(idx)
    patches.append(points)

downpcd_fps.paint_uniform_color([1.0, 0.0, 0.0])
geoms = [downpcd_fps] + patches

o3d.visualization.draw_geometries(geoms,
                                  zoom=0.3412,
                                  front=[0.4257, -0.2125, -0.8795],
                                  lookat=[2.6172, 2.0475, 1.532],
                                  up=[-0.0694, -0.9768, 0.2024], width=960, height=480)

