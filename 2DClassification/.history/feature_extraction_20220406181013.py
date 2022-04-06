import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import cv2

from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.feature import greycomatrix, greycoprops
from sklearn.feature_extraction import image as skimage


class Feature_Extractor():
    
    def __init__(self, path = "", data_path = "./range_images/", filename = "", hog_glcm = False, glcm_window_size = 20, glcm_step_size = 300, orb = False):
        """Extract features using a variety of methods

        Args:
            path (string): folder that contains range images
            filename (string): filename of range images
            hog (bool, optional): histogram of gradients execution. Defaults to True.
            glcm (bool, optional): [description]. Graylevel Comatrix execution. Defaults to True.
            orb (bool, optional): [description]. ORB feature descriptor execution. Defaults to True.
        """
        
        self.filename = filename
        self.dir = path + filename
        self.data_path = data_path
        self.glcm_window_size = glcm_window_size
        self.glcm_step_size = glcm_step_size
        self.class_dict = {"cube": '1', "cylinder": '2', "car": '3', "sphere": '4'}

        if hog_glcm:
            self.features, self.labels = self.combine_features(self.data_path)

        if orb:
            self.create_orb_dataset(self.data_path)

    def __call__(self):

        print("Feature Extraction Performed")

    def get_image(self, directory):
        """get image as numpy array

        Args:
            directory (string): directory of the range image

        Returns:
            grey_image: grey scaled image as a numpy array
        """
        
        img = Image.open(directory)
        if img.ndim != 2:
            np_img = np.asarray(img)[:, :, :1]
        else:
            np_img = np.asarray(img)
        return np_img 
    
    def get_cv_image(self, directory):
        """get image in an opencv compatible format

        Args:
            directory (string): directory of range image

        Returns:
            grey_image: greyscaled image as opencv object
        """
        img = cv2.imread(directory)
        return img
    
    def get_features(self, file):
        
        hog_features = self.get_hog_features(self.get_image(file))
        # glcm_features = self.get_glcm_features(self.get_image(file))

        # img_features = np.hstack((hog_features, glcm_features))

        return hog_features
    
    def get_hog_features(self, img):
        """
        HOG feature extraction on edge detected image
        Args:
            img (numpy array): grey scaled image

        Returns:
            hog_features: numpy array of the hog features of the image
        """
        
        # get HOG features from greyscale image

        # remove black background
        grey_image = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

        # resize to standard scale
        grey_image = cv2.resize(img, (256, 256), cv2.INTER_NEAREST)

        hog_features = hog(grey_image, block_norm='L2-Hys', pixels_per_cell=(8,8))
        
        return hog_features
    
    def glcm_elements(self, img):
        """Extract glcm elements from an image, texture descriptor using (7,7) patches

        Args:
            img (numpy array): range image
            step_size (int, optional): Step size for patch selection. Used for performance limited applications. Defaults to 20.

        Returns:
            (tuple): returns contrast, dissimilarity, correlation, homogeneity and ASM 
        """
        contrast_arr = []
        dis_arr = []
        hom_arr = [] 
        asm_arr = []
        cor_arr =[]

        # generate patches of size 7 by 7
        total_patches = skimage.extract_patches_2d(img, (self.glcm_window_size, self.glcm_window_size))
        
        set_patches = total_patches[::self.glcm_step_size]
        
        for patch in set_patches:
                g = greycomatrix(patch, [5], [0, np.pi/2], levels=256)
                contrast_arr.append(greycoprops(g, 'contrast')[0,0]) 
                dis_arr.append(greycoprops(g, 'dissimilarity')[0,0])
                hom_arr.append(greycoprops(g, 'homogeneity')[0,0])
                asm_arr.append(greycoprops(g, 'ASM')[0,0])
                cor_arr.append(greycoprops(g, 'correlation')[0,0])

        return contrast_arr, dis_arr, hom_arr, asm_arr, cor_arr
    

    def get_glcm_features(self, img):
        """Compile glcm features in one numpy array

        Args:
            img (numpy array)

        Returns:
            [features]: [complied glcm features for the image]
        """
        
        norm_image = cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        norm_image = norm_image.astype(np.uint8)      
        contrast, dis, hom, asm, cor = self.glcm_elements(norm_image)
        features = np.hstack((contrast, dis, hom, asm, cor))
        
        return features
    
    def combine_features(self, data_path):

        path = data_path

        init_file = "car_0027_no_25_1_715.png"

        feature_size = self.get_features(path + init_file).shape[0]

        features = np.empty((0, feature_size), int)

        labels = []
        
        for file in os.listdir(path):
            
            filename = os.fsdecode(file)
            
            file = path + filename
            
            if filename not in [".DS_Store"]:

                object_class = filename.split('_')[0]
                labels.append(self.class_dict[object_class])
                
                features = np.vstack([features, self.get_features(file)])

        labels = np.asarray(labels)
                 
        return features, labels 
    
    def get_orb_features(self, img):
        """Generate features using the ORB descriptor

        Args:
            img (numpy array): range image

        Returns:
            [keypoints, descriptors]: keypoints and their descriptors 
        """
        # create orb object
        orb = cv2.ORB_create()
        kp, des = orb.detectAndCompute(img,None)
        return kp, des
    
    def create_orb_dataset(self, path):
        
        for file in os.listdir(path):
            
            filename = os.fsdecode(file)

            file_path = path + filename

            if filename not in [".DS_Store"]:

                _, des = self.get_orb_features(self.get_cv_image(file_path))

                if type(des) != type(None):    
                    np.savetxt('./orb_desc/' + filename.split('.')[0], des, fmt='%d')



if __name__ == "__main__":
    
    path = "./range_images/"

    feature_extractor = Feature_Extractor(data_path = path, hog_glcm = False, orb = True) 

