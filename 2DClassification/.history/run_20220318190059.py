import os

from range_image import range_image

directory_str = "./b_point_clouds/"

directory = os.fsencode(directory_str)

for file in os.listdir(directory):
    
    filename = os.fsdecode(file)

    obj_class = filename.split("_")[0]

    range_im = range_image(filename, obj_class = obj_class, dst_path= "./b_range_images/", bg_rm= True, path = directory_str)
    