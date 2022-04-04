import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append("./Mask_RCNN-TF2")
from Mask_RCNN-TF2.ri_dataset 

import cv2


class background_subtractor():
    
    def __init__(self, image, blurred_image = None, filename = ''):
        
        self.image = image
        self.blurred_image = blurred_image
        self.filename = filename
        self.min_area = 0.0005
        self.max_area = 0.90
        self.dilate_iter = 5
        self.erode_iter = 5
        self.mask_color = (0.0)
        self.masked_image = self.background_subtraction(self.image, self.blurred_image)
    
    def extract_edges(self, image, blurred_image):
        
        edges = cv2.Canny(blurred_image, 0.1*np.iinfo(image.dtype).max, 0.6*np.iinfo(image.dtype).max, L2gradient = True).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))

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

        return edges
    
    def edges_output(self, mask, image):
        
        mask = mask.astype('float32') / 255.0
        image = image.astype('float32') / 255.0
        masked = np.multiply(mask, image) + (1 - mask)*self.mask_color
        masked_image = (masked * 255).astype(np.uint8)

        return masked_image

    def basic_background_subtraction(self, image, blurred_image):
    
        mask = self.extract_edges(image, blurred_image)
        
        mask[mask > 0] = cv2.GC_PR_FGD
        mask[mask == 0] = cv2.GC_PR_BGD
        
        outputMask = self._grabcut(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), mask)


        output = cv2.bitwise_and(image, image, mask=outputMask)

        return outputMask
    
    def ml_background_subtraction():
        
        
    
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

    def _grabcut(self, image, mask):
        
        fgModel = np.zeros((1, 65), dtype="float")
        bgModel = np.zeros((1, 65), dtype="float")  

        (mask, bgModel, fgModel) = cv2.grabCut(image, mask, None, bgModel, fgModel, iterCount=2, mode=cv2.GC_INIT_WITH_MASK)
        
        mask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1)

        outputMask = (mask * 255).astype("uint8")

        return outputMask



    