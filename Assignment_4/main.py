import numpy as np
import open3d as o3d
import os
import cv2

print("Load a ply point cloud, print and render it")
pcd = o3d.io.read_point_cloud("integrated.ply")
o3d.visualization.draw_geometries([pcd])