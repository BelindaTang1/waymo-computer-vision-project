import random
from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
import data_produce2
import cv2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from detectron2 import model_zoo


if __name__ == "__main__":
    cfg = get_cfg()
    
    cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'))
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    print('loading from: {}'.format(cfg.MODEL.WEIGHTS))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8   
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
    cfg.MODEL.DEVICE = "cuda"
    predictor = DefaultPredictor(cfg)
    
    for root, dirs, files in os.walk(r"/home/guanzhong/anaconda3/envs/Waymo/waymo-od/tutorial/agri_data/TrainDataSet_V2.2.3_ADD/background"): # Revise file path accordingly!!
        for file in files:
            data_f = os.path.join(root, file)
            im = cv2.imread(data_f)
            outputs = predictor(im)
            print("output is ", outputs)
            v = Visualizer(im[:, :, ::-1],
                        scale=0.8,
                        instance_mode=ColorMode.IMAGE_BW   
                        )
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            img = v.get_image()[:, :, ::-1]
            cv2.namedWindow('rr', cv2.WINDOW_NORMAL)
            cv2.imshow('rr', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()