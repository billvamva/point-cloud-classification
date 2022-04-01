# split into train and test set
import os
from re import M
from xml.etree import ElementTree
import numpy as np
from matplotlib import pyplot
from matplotlib.patches import Rectangle

from mrcnn.utils import Dataset
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import compute_ap
from mrcnn.model import load_image_gt
from mrcnn.model import mold_image

from tensorflow.keras.backend import manual_variable_initialization 
import tensorflow.compat.v1 as tf

manual_variable_initialization(True)



class ri_Dataset(Dataset):

    def __init__(self, mapping):

        self.mapping = mapping
        super().__init__()

    @classmethod
    def map_ids(self):

        mapping = {}

        num = 0

        for filename in os.listdir("../range_images/"):
            name = os.fsdecode(filename)

            if name == ".DS_Store":
                continue
            mapping[filename[:-4]] = str(num)
            num += 1

        return mapping


    def load_dataset(self, image_dir, annot_dir, train = True):

        self.add_class("dataset", 1, "object")

        for filename in os.listdir(image_dir):

            name = os.fsdecode(filename)

            if name == ".DS_Store":
                continue

            image_id = self.mapping[filename[:-4]]

            if int(image_id) in []:

                continue

            if train and int(image_id) >= 850:

                continue

            if not train and int(image_id) < 850:

                continue

            img_path = image_dir + filename

            annot_path = annot_dir + filename[:-4] + ".xml"

            self.add_image("dataset", image_id = image_id, path = img_path, annotation = annot_path)



    def find_box(self, filename):

        tree = ElementTree.parse(filename)

        root = tree.getroot()

        box = root.findall(".//bndbox")[0]

        xmin, xmax, ymin, ymax = int(box.find("xmin").text), int(box.find("xmax").text), int(box.find("ymin").text), int(box.find("ymax").text)

        coors = [xmin, ymin, xmax, ymax]

        width = int(root.find('.//size/width').text)

        height = int(root.find('.//size/height').text)

        return [coors], width, height

    def load_mask(self, image_id):

        img_str = ""

        for im_str, im_id in self.mapping.items():

           if int(image_id) == int(im_id):

               img_str = im_str

               break

        key = self.mapping[img_str]

        info = self.image_info[int(key)]

        path = info["annotation"]

        boxes, width, height = self.find_box(path)

        mask = np.zeros([height, width, 1], dtype = 'uint8')
        
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            mask[row_s:row_e, col_s:col_e, i] = 1

        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    # load an image reference
    def image_reference(self, image_id):

        info = self.image_info[str(image_id)]
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
    NUM_CLASSES = 1 + 1
    BATCH_SIZE = 2
    IMAGE_MIN_DIM = 480
    IMAGE_MAX_DIM = 640
    RPN_ANCHOR_SCALES = (16, 32, 48, 64, 128)
    RPN_ANCHOR_RATIOS = [0.5, 1, 1.5]
    TRAIN_ROIS_PER_IMAGE = 20
    DETECTION_MIN_CONFIDENCE = 0.5
    STEPS_PER_EPOCH = 850

# define the prediction configuration
class PredictionConfig(Config):
# define the name of the configuration
    NAME = "ri_test_cfg"
    USE_MINI_MASK = False
    NUM_CLASSES = 1 + 1
    # simplify GPU config
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

# calculate the mAP for a model on a given dataset
def evaluate_model(dataset, model, cfg):
	APs = list()
	for image_id in dataset.image_ids:
		# load image, bounding boxes and masks for the image id
		image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(dataset, cfg, image_id)
		# convert pixel values (e.g. center)
		scaled_image = mold_image(image, cfg)
		# convert image into one sample
		sample = np.expand_dims(scaled_image, 0)
		# make prediction
		yhat = model.detect(sample, verbose=0)
		# extract results for first sample
		r = yhat[0]
		# calculate statistics, including AP
		AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r["rois"], r["class_ids"], r["scores"], r['masks'])
		# store
		APs.append(AP)
	# calculate the mean AP across all images
	mAP = np.mean(APs)
	return mAP

# plot a number of photos with ground truth and predictions
def plot_actual_vs_predicted(dataset, model, cfg, n_images=7):
    # load image and mask
    for i in range(n_images):
        # load the image and mask
        image = dataset.load_image(i)
        mask, _ = dataset.load_mask(i)
        # convert pixel values (e.g. center)
        scaled_image = mold_image(image, cfg)
        # convert image into one sample
        sample = np.expand_dims(scaled_image, 0)
        # make prediction
        yhat = model.detect(sample, verbose=0)[0]
        print(yhat['rois'])
        # define subplot
        pyplot.subplot(n_images, 2, i*2+1)
        # plot raw pixel data
        pyplot.imshow(image)
        pyplot.title('Actual')
        # plot masks
        for j in range(mask.shape[2]):
            pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
        # get the context for drawing boxes
        pyplot.subplot(n_images, 2, i*2+2)
        # plot raw pixel data
        pyplot.imshow(image)
        pyplot.title('Predicted')
        ax = pyplot.gca()
        # plot each box
        for box in yhat['rois']:
            # get coordinates
            y1, x1, y2, x2 = box
            # calculate width and height of the box
            width, height = x2 - x1, y2 - y1
            # create the shape
            rect = Rectangle((x1, y1), width, height, fill=False, color='red')
            # draw the box
            ax.add_patch(rect)
            # show the figure
    #pyplot.show()

def test_mask_load(train_set):

    for i in range(9):
    	# define subplot
        pyplot.subplot(330 + 1 + i)
        # plot raw pixel data
        image = train_set.load_image(i)
        pyplot.imshow(image)
        # plot all masks
        mask, _ = train_set.load_mask(i)
        for j in range(mask.shape[2]):
            pyplot.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
        # show the figure
    pyplot.show()

mapping = ri_Dataset.map_ids()

train_set = ri_Dataset(mapping)
train_set.load_dataset("../range_images/", "../range_images_anot/")
train_set.prepare()
print('Train: %d' % len(train_set.image_ids))


test_set = ri_Dataset(mapping)
test_set.load_dataset("../range_images/", "../range_images_anot/", train = False)
test_set.prepare()
print('Test: %d' % len(test_set.image_ids))


######################################################################################
# config = ri_config()
# config.display()
# model = MaskRCNN(mode='training', model_dir='./', config=config)
# # model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
# model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs = 15, layers='all')
######################################################################################

######################################################################################
cfg = PredictionConfig()
# define the model
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
# load model weights
model.load_weights('mask_rcnn_ri_cfg_0015.h5', by_name=True)
tf.keras.Model.load_weights(model.keras_model, 'mask_rcnn_ri_cfg_0015.h5', by_name=True)
######################################################################################

######################################################################################
# # evaluate model on training dataset
# train_mAP = evaluate_model(train_set, model, cfg)
# print("Train mAP: %.3f" % train_mAP)
# # evaluate model on test dataset
# test_mAP = evaluate_model(test_set, model, cfg)
# print("Test mAP: %.3f" % test_mAP)
######################################################################################

######################################################################################
# plot predictions for train dataset
plot_actual_vs_predicted(train_set, model, cfg)
# plot predictions for test dataset
plot_actual_vs_predicted(test_set, model, cfg)
######################################################################################


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