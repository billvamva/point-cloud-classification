import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import open3d as o3d
import scipy.interpolate as interp
import os
import itertools

class range_image():
    
    def __init__(self, filename, vox = 0.01, path = ""):
        """
        Range Image Construction Class

        Args:
            method (string): Select Range Method Construction Methodology
            filename (string): Filename of PointCloud
            vox (float, optional): [description]. Used for Voxel Downsampling. Defaults to 0.01.
            path (str, optional): [description]. Path where point clouds are stored. Defaults to "".
        """

        self.filename = filename
        self.pcd = self.open_pcd(filename, path)
        self.vox = vox
        self.o3d_pcd, self.xyz_load = self.op3d(self.pcd, self.vox)
        self.range_data = self.range_image_pbea(self.xyz_load)

    
    def open_pcd(self, filename, path = ""):
        
        pcd = o3d.io.read_point_cloud(path + filename)

        return pcd

    def op3d(self, pcd, vox):
        """Convert to open3d point cloud

        Args:
            pcd (pcd format): .pcd point cloud taken from matlab
            vox (float): determines the downsampling of the point cloud

        Returns:
            o3d_pcd: open3d point cloud
            xyz_load: numpy array of point cloud
        """
        o3d_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(pcd.points)))

        o3d_pcd = o3d_pcd.voxel_down_sample(voxel_size=vox)

        xyz_load = np.asarray(o3d_pcd.points)

        return o3d_pcd, xyz_load
    
    def plot_3d(self):
        """Point Cloud Visualization
        """
        o3d.visualization.draw_geometries([self.o3d_pcd])
    
    def center_pcd(self, points, off):
        """Centers Point Cloud using means of each dimension

        Args:
            points (numpy array): point cloud in numpy array format

        Returns:
            points (numpy array): centered point cloud
        """

        points[:, 0] = points[:, 0] + np.mean(points[:, 0])

        points[:, 1] = points[:, 1] + off[0] * np.mean(points[:, 1])  

        points[:, 2] = points[:, 2] + off[1] * np.mean(points[:, 2])

        return points
    
    def normalize_pcd(self, data):
        data[:, 0] = (data[:, 0] - np.min(data[:, 0])) / (np.max(data[:, 0]) - np.min(data[:, 0]))
        data[:, 1] = (data[:, 1] - np.min(data[:, 1])) / (np.max(data[:, 1]) - np.min(data[:, 1]))
        data[:, 2] = (data[:, 2] - np.min(data[:, 2])) / (np.max(data[:, 2]) - np.min(data[:, 2]))

        return 100*data
    
    def shift_range(self,arr):
        arr = (2*np.pi + arr) * (arr < 0) + arr*(arr > 0)        
        return arr
        

    # Range Image construction
    def range_image_pbea(self, points):
        """"
        Construction of numpy array

        Args:
            points (numpy array): the point cloud in matrix form 

        Returns:
            image (numpy array): generated image in matrix form
        """

        offset_permutations = itertools.product(range(-1, 2), repeat=2)

        for off in offset_permutations:
            
            points = self.center_pcd(self.normalize_pcd(points), off)

            # points = self.normalize_pcd(self.center_pcd(points, off))

            x = points[:, 1]
            y = points[:, 0]
            z = points[:, 2]
            
            r = np.sqrt(np.square(x) + np.square(y) + np.square(z))

            phi = np.arctan(np.divide(x, (y + 1**-10)))
            
            theta = np.arcsin(np.divide(z, r))
            
            u = np.floor(1/2*(1+(phi/np.pi))*512)
            
            col_start, col_end = (int(min(u)), int(max(u) + 1))

            v = np.floor((np.max(theta)-theta)/(np.max(theta)-np.min(theta) + 1**-10)*424)

            row_start, row_end = (int(min(v)), int(max(v) + 1))
            
            image = np.zeros((v.shape[0], u.shape[0]))
            
            image_dict = {}

            for i, j, value in zip(v, u, r):
                
                if (i,j) in image_dict and image_dict[(int(i), int(j))] >= value:
                    continue
                
                else:
                    image[int(i), int(j)] = value
                    image_dict[(int(i), int(j))] = value
            
            range_data =  image[row_start:row_end, col_start:col_end]

            self.image = self.form_img(range_data)

            self.save_image(off)

    def form_img(self, arr):
        
        """Interpolates image values to [0, 255] range

        Returns:
            image: manipulated image that can be displayed
        """
        
        r, c = arr.shape
        
        flat_arr = arr.flatten()

        mapping = interp.interp1d([min(flat_arr), max(flat_arr)], [0, 255])

        interp_arr = mapping(flat_arr)
        
        new_arr = interp_arr.reshape(r, c)
        
        return new_arr

    def background_subtraction(self, points, dist):
        dist_mean = np.mean(dist)
        points = points[np.where(dist < dist_mean)]
        dist = dist[np.where(dist < dist_mean)]
        return points, dist  
     
    def show_image(self):
        """Range Image Visualization
        """
        plt.imshow(self.image)
        plt.show()
    
    def save_image(self, pos):
        """Saves image to local directory
        """
        path = "./range_images/"

        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)    
        ax.imshow(self.image, aspect='auto')
            
        fig.savefig(f"{path}{self.filename.split('.')[0]}_{pos[0]}{pos[1]}.png")

if __name__ == "__main__":
    
    directory_str = "./point_clouds/"

    directory = os.fsencode(directory_str)

    for file in os.listdir(directory):
        
        filename = os.fsdecode(file)

        range_im = range_image(filename, path = directory_str)
    