
import random
from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
import data_register
from cv2 import cv2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from detectron2.evaluation import SemSegEvaluator
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

test_dataset = 'train'
MetadataCatalog.get(test_dataset).thing_classes = ['cell']
fruits_nuts_metadata = MetadataCatalog.get(test_dataset)


if __name__ == "__main__":

    cfg = get_cfg()
    cfg.merge_from_file(
        "configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.MODEL.WEIGHTS = "./output/model_1500.pth"
    print('loading from: {}'.format(cfg.MODEL.WEIGHTS))
    # set the testing threshold for this model
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
    cfg.DATASETS.TEST = (test_dataset, )
    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator(
        test_dataset, ("bbox", "segm"), False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, test_dataset, num_workers=1)
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

    data_f = 'data/train/TCGA-18-5592-01Z-00-DX1.tif'
    im = cv2.imread(data_f)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=fruits_nuts_metadata,
                   scale=1.5,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
                   )
    p = outputs["instances"].to("cpu")
    vis_output = v.draw_instance_predictions(p)
    img = vis_output.get_image()[:, :, ::-1]
    cv2.imshow('rr', img)
    cv2.waitKey(0)
    cv2.imwrite('./output/rr.jpg', img)
