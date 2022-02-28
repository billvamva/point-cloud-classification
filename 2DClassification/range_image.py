import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import scipy.interpolate as interp
from scipy.linalg import lstsq
import os
import cv2
from skimage.filters import hessian

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
        self.incline = 0
        self.blur = 21
        self.min_area = 0.0005
        self.max_area = 0.90
        self.dilate_iter = 10
        self.erode_iter = 10
        self.mask_color = (0.0)
        self.o3d_pcd, self.xyz_load = self.op3d(self.pcd, self.vox)
        self.plane_params = self.plane_projection(self.xyz_load)
        self.s_xyz_load = self.shift_plane(self.plane_params, self.xyz_load)
        self.range_data, self.s_range_data = self.range_image_pbea(self.xyz_load), self.range_image_pbea(self.s_xyz_load)
        self.image, _ = self.form_img(self.range_data) 
        _, self.s_blurred_image = self.form_img(self.s_range_data)
        # self.cropped_image = self.background_subtraction(self.image, self.s_blurred_image)
        self.cropped_image = self.image
        self.save_image()

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

        blurred_image = cv2.GaussianBlur(image, (7, 7), sigmaX=33, sigmaY=33)

        return image, blurred_image
    
    def extract_channel(self, image):
        bgr_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(hsv)

        return h

    def background_subtraction(self, image, blurred_image):

        edges = cv2.Canny(blurred_image, 0.1*np.iinfo(image.dtype).max, 0.6*np.iinfo(image.dtype).max, L2gradient = True).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5))

        kernel = None

        edges = cv2.dilate(edges, kernel)

        contour_info = [(c, cv2.contourArea(c)) for c in cv2.findContours(edges.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[1]]

        for contour, _ in contour_info:

            for i, _ in enumerate(contour):

                contour[i] = cv2.convexHull(contour[i])
        
        self.save_contour(contour_info, image)
        
        image_area = image.shape[0] * image.shape[1]

        min_area = self.min_area * image_area
        
        max_area = self.max_area * image_area

        mask = np.zeros(image.shape, dtype = np.uint8)

        for contour in contour_info:
            
            if contour[1] > min_area and contour[1] < max_area:
                
                mask = cv2.fillConvexPoly(mask, contour[0], 255)
                
        mask = cv2.dilate(mask, kernel, iterations=self.dilate_iter)
        mask = cv2.erode(mask, kernel, iterations=self.erode_iter)

        mask = mask.astype('float32') / 255.0
        image = image.astype('float32') / 255.0

        masked = np.multiply(mask, image) + (1 - mask)*self.mask_color

        masked_image = (masked * 255).astype(np.uint8)
        
        return masked_image
    
    def save_contour(self, contour_info, image):
        
        contour_image = np.zeros((image.shape[0], image.shape[1], 3))

        cv2.drawContours(contour_image, list(zip(*contour_info))[0], -1, (0,255,0), 3)

        scale_percent = 220 # percent of original size
        width = int(contour_image.shape[1] * scale_percent / 100)
        height = int(contour_image.shape[0] * scale_percent / 100)
        dim = (width, height)
        
        # resize image
        contour_image = cv2.resize(contour_image, dim, interpolation = cv2.INTER_AREA)

        cv2.imwrite(f"./contours/{self.filename.split('.')[0]}.png", contour_image)
     
    def show_image(self):
        """Range Image Visualization
        """
        plt.imshow(self.image)
        plt.show()
    
    def save_image(self):
        """Saves image to local directory
        """
        path = "./range_images/"

        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)    
        ax.imshow(self.cropped_image, aspect='auto')
            
        fig.savefig(f"{path}{self.filename.split('.')[0]}.png")
        plt.close(fig)

if __name__ == "__main__":
    
    directory_str = "./point_clouds/"
    
    # filename = "block_1.pcd"
    
    # range_im = range_image(filename, path = directory_str)

    directory = os.fsencode(directory_str)

    for file in os.listdir(directory):
        
        filename = os.fsdecode(file)

        range_im = range_image(filename, path = directory_str)
    