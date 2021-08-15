# link to json_produce.py
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets import load_coco_json
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
import cv2
from detectron2.utils.visualizer import Visualizer
import random

register_coco_instances("waymodata_train", {}, "./json_data/train_record.json","./frame_imgs/total_train")

register_coco_instances("waymodata_val", {}, "./json_data/val_record.json","./frame_imgs/total_val")