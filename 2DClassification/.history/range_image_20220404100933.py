import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import scipy.interpolate as interp
from scipy.linalg import lstsq
import os
import cv2
import time

from background_subtractor import background_subtractor

class range_image():
    
    def __init__(self, filename, obj_class = None, dst_path = "./nobg_range_images/",vox = 0.01, path = "", bg_rm = False):
        """
        Range Image Construction Class

        Args:
            method (string): Select Range Method Construction Methodology
            filename (string): Filename of PointCloud
            vox (float, optional): [description]. Used for Voxel Downsampling. Defaults to 0.01.
            path (str, optional): [description]. Path where point clouds are stored. Defaults to "".
        """

        self.filename = filename
        self.obj_class = obj_class
        self.path = path
        self.pcd = self.open_pcd(filename, path)
        self.dst_path = dst_path
        self.vox = vox
        self.bg_rm = bg_rm
        self.incline = 0
        self.blur = 21
        self.o3d_pcd, self.xyz_load = self.op3d(self.pcd, self.vox)
        self.plane_params = self.plane_projection(self.xyz_load)
        self.s_xyz_load = self.shift_plane(self.plane_params, self.xyz_load)
        self.range_data, self.s_range_data = self.range_image_pbea(self.xyz_load), self.range_image_pbea(self.s_xyz_load)

        if self.range_data.size == 0:
            return 

        self.image, self.blurred_image = self.form_img(self.range_data) 
        self.s_image, self.s_blurred_image = self.form_img(self.s_range_data)
        
        if self.bg_rm:
            self.background_subtractor = background_subtractor(self.image)
            self.cropped_image = self.background_subtractor.masked_image
        
        else:
            self.cropped_image = self.image

        self.output_path = self.save_image(self.dst_path)

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
    
    def translate_data(self, data, lb, ub):

        data_t = np.zeros_like(data)
        
        for i in range(data.shape[1]):
                 
            translate = interp.interp1d([min(data[:, i]), max(data[:, i])], [lb, ub])

            data_t[:, i] = translate(data[:, i])

        return data_t
    
    def plane_projection(self, data):

        
        A = np.c_[data[:, 0], data[:, 1], np.ones_like(data[:, 0])]
        b = np.transpose(data[:, 2])
        
        fit, residual, _, _ = lstsq(A, b)

        print("solution: %f x + %f y + %f = z" % (fit[0], fit[1], fit[2]))
        print("residual:", residual)
        
        return fit
    
    def shift_plane(self, fit, data):
        
        data_t = np.zeros_like(data)        

        theta = ((self.incline) * np.pi) / 180

        trans_matrix = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

        s_normal = np.matmul(trans_matrix, np.transpose([fit[0], fit[1], 1])) 

        plane_point = [1, 1, np.sum(fit)]
        
        for i in range(data.shape[0]):
            
            v = data[i, :] - plane_point
            
            dist = np.dot(s_normal, v)

            data_t[i, :]  = data[i, :] - dist*s_normal

        return data_t
         
    def plot_3d(self):
        """Point Cloud Visualization
        """
        o3d.visualization.draw_geometries([self.o3d_pcd])
    
    def plot_mtlb(self):
        
        plt.figure()

        data_t = self.translate_data(self.xyz_load, 0, 1)
        
        ax = plt.subplot(projection = '3d')
        # ax.scatter(data_t[0], data_t[1], data_t[2], color = 'b')
        # plot plane
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        X,Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
                        np.arange(ylim[0], ylim[1]))
        Z = np.zeros(X.shape)

        fit = self.plane_projection(data_t)

        for r in range(X.shape[0]):
            for c in range(X.shape[1]):
                Z[r,c] = fit[0] * X[r,c] + fit[1] * Y[r,c] + fit[2]

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.plot_wireframe(X,Y,Z, color='k')
        
        plt.show()

    def shift_pcd(self, points):
        """Shifts Point Cloud using means of each dimension

        Args:
            points (numpy array): point cloud in numpy array format

        Returns:
            points (numpy array): shifted point cloud
        """

        points[:, 0] = points[:, 0]

        points[:, 1] = points[:, 1]

        points[:, 2] = points[:, 2]

        return points
    
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

            
        points = self.shift_pcd(points)

        if np.isnan(points).all():

            return np.array([]) 

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

        return range_data

    def form_img(self, arr):
        
        """Interpolates image values to [0, 255] range

        Returns:
            image: manipulated image that can be displayed
        """
        
        r, c = arr.shape
        
        flat_arr = arr.flatten()

        mapping = interp.interp1d([min(flat_arr), max(flat_arr)], [0, 256])

        interp_arr = mapping(flat_arr)
        
        new_arr = interp_arr.reshape(r, c).astype(np.uint8)
        
        image, blurred_image = self.preprocess_image(new_arr)

        return image, blurred_image
    
    def preprocess_image(self, image):
        """Sharpen Image using 2D filter

        Args:
            image (array): input image

        Returns:
            array: filtered output image
        """
        
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        
        image = cv2.filter2D(image, -1, kernel)

        blurred_image = cv2.GaussianBlur(image, (5,5), cv2.BORDER_DEFAULT)

        return image, blurred_image
    
    def extract_channel(self, image):
        bgr_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)

        return h


    def show_image(self):
        """Range Image Visualization
        """
        plt.imshow(self.image)
        plt.show()
    
    def save_image(self, dst_path):
        """Saves image to local directory
        """

        els = self.path.split("_")

        if self.obj_class == 'car':
            
            output_path = f"{dst_path}{self.obj_class}_{els[2]}_{self.filename.split('.')[0]}.png" 
        
        elif self.obj_class:
            
            output_path = f"{dst_path}{self.obj_class}_{self.filename.split('.')[0]}.png"           
        
        else:

            output_path = dst_path + time.strftime("%Y%m%d-%H%M%S") + ".png"
            
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)    
        ax.imshow(self.cropped_image, aspect='auto')
            
        fig.savefig(output_path)
        plt.close(fig)

        return output_path

if __name__ == "__main__":
    
    directory_str = "./point_clouds/"
    
    directory = os.fsencode(directory_str)

    for folder in os.listdir(directory):

        folder_str = os.fsdecode(folder)

        if folder_str == ".DS_Store":
            
            continue
        
        folder_str = folder_str + "/" 

        obj_class = folder_str.split('_')[0]
        
        print(obj_class)
        
        for file in os.listdir(os.path.join(directory,folder)): 
            
            filename = os.fsdecode(file)

            print(filename.split(".")[-1])

            if filename.split('.')[-1] == "pcd":
                
                range_im = range_image(filename, obj_class = obj_class, path = directory_str + folder_str, fold = folder_str, bg_rm = True)
            
    