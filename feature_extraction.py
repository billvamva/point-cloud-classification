import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
import cv2

from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.feature import greycomatrix, greycoprops
from skimage.filters import hessian, median


class Feature_Extractor():
    
    def __init__(self, path, filename, hog = True, glcm = True, orb = True):
        
        self.filename = filename
        self.dir = path + filename
        self.hog = hog
        self.glcm = glcm
        if hog:
            self.hog_features = self.get_hog_features(self.get_image(self.dir))
        if glcm:
            self.glcm_features = self.get_glcm_features(self.get_image(self.dir))
        if orb:
            self.orb_kps, self.orb_des = self.get_orb_features(self.get_cv_image(self.dir))
            self.matches = self.match_orb_features(self.orb_des)

    def __call__(self):

        print("Feature Extraction Performed")

    def get_image(self, directory):   
        
        img = Image.open(directory)
        np_img = np.asarray(img)
        print(np_img.shape)
        grey_image = rgb2gray(np_img)
        return grey_image 
    
    def get_cv_image(self, directory):
        
        img = cv2.imread(directory)
        grey_image= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        return grey_image
    
    def get_hog_features(self, img):
        
        # pass through hessian filter for edge detection
        edge_detected = hessian(img, mode = 'constant')

        # get HOG features from greyscale image
        hog_features = hog(edge_detected, block_norm='L2-Hys', pixels_per_cell=(64, 64))
        
        return hog_features
    
    def glcm_elements(self, img, step_size = 20):
        contrast_arr = []
        dis_arr = []
        hom_arr = [] 
        asm_arr = []
        cor_arr =[]

        # generate patches of size 7 by 7
        total_patches = np.lib.stride_tricks.sliding_window_view(img, (7,7))
        
        set_patches = total_patches[::step_size]
        
        for patches in set_patches:
            for patch in patches:
                # glcm features
                g = greycomatrix(patch, [5], [0, np.pi/2], levels=256)
                contrast_arr.append(greycoprops(g, 'contrast')[0,0]) 
                dis_arr.append(greycoprops(g, 'dissimilarity')[0,0])
                hom_arr.append(greycoprops(g, 'homogeneity')[0,0])
                asm_arr.append(greycoprops(g, 'ASM')[0,0])
                cor_arr.append(greycoprops(g, 'correlation')[0,0])

        return contrast_arr, dis_arr, hom_arr, asm_arr, cor_arr

    def get_glcm_features(self, img):
        
        norm_image = cv2.normalize(img, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
        norm_image = norm_image.astype(np.uint8)      
        contrast, dis, hom, asm, cor = self.glcm_elements(norm_image)
        features = np.hstack((contrast, dis, hom, asm, cor))
        
        return features
    
    def get_orb_features(self, img):
        """[summary]

        Args:
            img ([type]): [description]

        Returns:
            [type]: [description]
        """
        # create orb object
        orb = cv2.ORB_create()
        kp, des = orb.detectAndCompute(img,None)
        np.savetxt('./orb_desc/' + ''.join(self.filename.split('_')[:2]), des, fmt='%d')
        return kp, des
    
    def match_orb_features(self, des1):
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True) 
        
        directory_str = "./orb_desc/"

        directory = os.fsencode(directory_str)

        matches_num = {}

        for file in os.listdir(directory):
            
            filename = os.fsdecode(file)

            des2 = np.loadtxt(directory_str + filename, dtype=np.uint8)
            matches = bf.match(des1, des2)

            matches = sorted(matches, key = lambda x:x.distance)

            matches_num[filename] = len(matches)
            
        return matches_num

if __name__ == "__main__":

    path = "./range_images/"
    filename = "car_1_range_image_pbea.png"

    feature_extractor = Feature_Extractor(path, filename, hog = False, glcm= False)

    print(feature_extractor.matches)
