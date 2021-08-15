from operator import mod
from detectron2 import checkpoint
from detectron2.data.catalog import DatasetCatalog
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.evaluation import DatasetEvaluator, evaluator
import torch
from torch.utils.data import dataloader
import data_produce
from detectron2.config import get_cfg
from detectron2 import model_zoo
import os
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader

class Counter(DatasetEvaluator):

    def reset(self):
        self.count = 0
    def process(self, inputs, outputs):
        for output in outputs:
            self.count += len(output['instances'])
    def evaluate(self):
        return {"count": self.count}

def get_all_inputs_outputs(data_loader, model):
    for data in data_loader:
        model.eval()
        with torch.no_grad():
            #print(model(data))
            print(len(model(data)[0]['instances']))
            yield data, model(data)


cfg = get_cfg()
    
cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'))
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
print('loading from: {}'.format(cfg.MODEL.WEIGHTS))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8   
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5
cfg.DATASETS.TEST = ("waymodata_val", )
cfg.MODEL.DEVICE = "cuda"

data_loader = build_detection_test_loader(DatasetCatalog.get("waymodata_val"), mapper=DatasetMapper(cfg, is_train=False))

model = build_model(cfg)
checkpoint = DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

evaluator = Counter()
evaluator.reset()
for inputs, outputs in get_all_inputs_outputs(data_loader, model):
    evaluator.process(inputs, outputs)
eval_results = evaluator.evaluate()

