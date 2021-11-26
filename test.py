import numpy as np
import open3d as o3d
import sys

def plot_pcd(path):
    
    bin_pcd = np.fromfile(path , np.float32)

    # Reshape and drop reflection values
    points = bin_pcd.reshape((-1, 3))[:, 0:3]

    # Convert to Open3D point cloud
    o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

    o3d.visualization.draw_geometries([o3d_pcd]) 

if __name__ == '__main__':
    
    path = sys.argv[1]

    plot_pcd(path)
