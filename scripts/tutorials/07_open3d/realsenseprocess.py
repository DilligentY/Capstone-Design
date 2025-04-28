import pyrealsense2 as rs
import open3d as o3d
import numpy as np


def RANSAC_Plane_Segmentation(pcd : o3d.geometry.PointCloud, dist_thr = 0.01, ransac_n=3, num_iterations=1000) -> tuple[o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    plane_model, inlier = pcd.segment_plane(dist_thr, ransac_n, num_iterations)
    
    pcd_in = pcd.select_by_index(inlier)
    pcd_out = pcd.select_by_index(inlier, invert=True)
    
    return pcd_in, pcd_out
    
    
def Voxel_Downsampling(pcd : o3d.geometry.PointCloud, x = 0.01) -> o3d.geometry.PointCloud:
    
    return pcd.voxel_down_sample(x)
    

def FPS_Dowsampling(pcd : o3d.geometry.PointCloud, num_samples = 100) -> o3d.geometry.PointCloud:
    
    return pcd.farthest_point_down_sample(num_samples)

    
def KD_Tree(pcd : o3d.geometry.PointCloud) -> o3d.geometry.KDTreeFlann:
    
    return o3d.geometry.KDTreeFlann(pcd)


def KNN_Search(pcd : o3d.geometry.PointCloud, kd_tree : o3d.geometry.KDTreeFlann, num_iterations=100) -> list:
    patches = list()
    
    for i in range(num_iterations):
        [k, idx, _] = kd_tree.search_knn_vector_3d(pcd.points[i], 50)
        points = pcd.select_by_index(idx)
        patches.append(points)
    
    return patches
