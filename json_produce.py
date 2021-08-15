import numpy as np
import pandas as pd
import json
import random
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
from detectron2.structures import BoxMode
import os

tf.compat.v1.enable_eager_execution() 

####### Produce COCO format

def my_dataset_function(tfr_dir, mode, img_dir):
    json_dict = {'images': [], 
                'categories': [{'supercategory': 'unknown', 'id': 1, 'name': 'unknown'},
                                {'supercategory': 'vehicle', 'id': 2, 'name': 'vehicle'},
                                {'supercategory': 'pedestrian', 'id': 3, 'name': 'pedestrian'},
                                {'supercategory': 'sign', 'id': 4, 'name': 'sign'},
                                {'supercategory': 'cyclist', 'id': 5, 'name': 'cyclist'}],
                'annotations': []}

    #folder_num = 0
    count = 0
    img_dir = os.path.join(img_dir, "totol_" + mode)
    print(img_dir)
    img_num = 0

    # image
    for root, _, files in os.walk(os.path.join(tfr_dir, mode)):
        for file in files: 
            FILENAME = os.path.join(root, file)
            dataset = tf.data.TFRecordDataset(FILENAME, compression_type='')
            for data in dataset: 
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(data.numpy()))

                for index, camera_image in enumerate(frame.images): # frame by frame
                    img_numpy = tf.image.decode_jpeg(camera_image.image).numpy()
                    img_width = img_numpy.shape[1]
                    img_height =img_numpy.shape[0]

                    json_dict['images'].append({'height': img_height,
                                                'width': img_width,
                                                'id': img_num,
                                                'file_name': str(img_num) + '.jpg'})

                    for camera_labels in frame.camera_labels:
                        if camera_labels.name != camera_image.name:
                            continue

                        for label in camera_labels.labels:
                            
                            json_dict['annotations'].append({#'image_id':str(folder_num) + " " + str(index),
                                                            'image_id': img_num,
                                                            'bbox': [label.box.center_x - 0.5 * label.box.length, label.box.center_y - 0.5 * label.box.width,
                                                            label.box.length, label.box.width],
                                                            'category_id': label.type + 1,
                                                            'iscrowd': 0,
                                                            'area': label.box.length * label.box.width,
                                                            'segmentation': [],
                                                            'id': count})
                                                            
                            count = count + 1
                    
                    img_num = img_num + 1 
                #folder_num =  folder_num + 1
    
    print(img_num)
    return json_dict



tfr_dir = r'/home/guanzhong/anaconda3/envs/Waymo/waymo-od/tutorial/tfrecords'
img_dir = "/home/guanzhong/anaconda3/envs/Waymo/waymo-od/tutorial/frame_imgs/"

train_json_dict = my_dataset_function(tfr_dir, 'train', img_dir)

train_json_str = json.dumps(train_json_dict, indent=4)
with open('./json_data/train_record.json', 'w') as f:
    f.write(train_json_str)
    print('process complete')


val_json_dict = my_dataset_function(tfr_dir, 'val', img_dir)

val_json_str = json.dumps(val_json_dict, indent=4)
with open('./json_data/val_record.json', 'w') as f:
    f.write(val_json_str)
    print('process complete')


