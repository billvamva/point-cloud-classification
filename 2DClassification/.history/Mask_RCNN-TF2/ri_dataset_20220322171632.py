# split into train and test set
import os
from re import M
from xml.etree import ElementTree
import numpy as np
from matplotlib import pyplot as plt

from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.visualize import display_instances
from mrcnn.utils import extract_bboxes
from mrcnn.model import MaskRCNN


class ri_Dataset(Dataset):

    def __init__(self, mapping):
        
        self.mapping = mapping
        super().__init__()

    @classmethod
    def map_ids(self):
        
        mapping = {}

        num = 0

        for filename in os.listdir("../range_images/"): 

            mapping[filename[:-4]] = num
            
            num += 1
        
        return mapping
        
        
    
    def load_dataset(self, image_dir, annot_dir, train = True):

        self.add_class("dataset", 1, "cube")
        self.add_class("dataset", 2, "cylinder")
        self.add_class("dataset", 3, "car_0027")
        self.add_class("dataset", 4, "car_0198")
        self.add_class("dataset", 5, "sphere")
        
        for filename in os.listdir(image_dir): 

            image_id = self.mapping[filename[:-4]] 

            if image_id in []:
                
                continue

            if train and image_id % 4 == 0:
                
                continue
                
            if not train and image_id % 4 != 0:
                
                continue
            
            img_path = image_dir + filename

            annot_path = annot_dir + filename[:-4] + ".xml"

            self.add_image("dataset", image_id = image_id, path = img_path, annotation = annot_path)

        
    
    def find_box(self, filename):
        
        tree = ElementTree.parse(filename)

        root = tree.getroot()

        box = root.findall(".//bndbox")[0]
        
        xmin, xmax, ymin, ymax = int(box.find("xmin").text), int(box.find("xmax").text), int(box.find("ymin").text), int(box.find("ymax").text)

        coors = [xmin, xmax, ymin, ymax]
        
        width = int(root.find('.//size/width').text)
        
        height = int(root.find('.//size/height').text)

        return coors, width, height

    def load_mask(self, image_id):

        for im_str, im_id in self.mapping.items():
            
           if image_id == im_id:
               
               image_str = im_str
               
               break 
            
        obj = image_str.split("_")[0]

        info = self.image_info[self.mapping[image_id]]

        path = info["annotation"]

        coors, width, height = self.find_box(path)

        mask = np.zeros([height, width, 1], dtype = 'uint8')

        row_s, row_e = coors[2], coors[3]

        col_s, col_e = coors[0], coors[1]

        mask[row_s:row_e, col_s:col_e, :] = 1

        class_id = self.class_names.index(obj)

        return mask, np.array([class_id]) 
    
    # load an image reference
    def image_reference(self, image_id):
        
        info = self.image_info[image_id]
        return info['path']
    
    def enumerate_ims(self):
        # enumerate all images in the dataset
        for image_id in self.image_ids:
            # load image info
            info = self.image_info[image_id]
            # display on the console
            print(info)
        

class ri_config(Config):
    
    NAME = "ri_cfg"

    NUM_CLASSES = 5

    STEPS_PER_EPOCH = 816

mapping = ri_Dataset.map_ids()

train_set = ri_Dataset(mapping)
train_set.load_dataset("../range_images/", "../range_images_anot/")
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))



test_set = ri_Dataset(mapping)
test_set.load_dataset("../range_images/", "../range_images_anot/", train = False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))


config = ri_config()
config.display()

model = MaskRCNN(mode='training', model_dir='./', config=config)

model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')



def change_xml_files():
    
    path = '../range_images_anot/'
    base_dst_path = "/Users/vasilieiosvamvakas/Documents/Project/2DClassification/range_images/"

    for file in os.listdir(path):

        filename = os.fsdecode(file) 

        if filename == ".DS_Store":
            continue

        dst_path = base_dst_path + filename 
        mytree = ElementTree.parse(path + filename)
        myroot = mytree.getroot()
        myroot[0].text = "range_images"
        myroot[2].text = dst_path  

        mytree.write(path + filename)