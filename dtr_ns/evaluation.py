
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

test_dataset = 'test'
MetadataCatalog.get(test_dataset).thing_classes = ['cell']
fruits_nuts_metadata = MetadataCatalog.get(test_dataset)


if __name__ == "__main__":
    cfg = get_cfg()
    cfg.merge_from_file(
        "configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
    cfg.DATALOADER.NUM_WORKERS = 1
    cfg.MODEL.DEVICE = 'cuda'
    # initialize from model zoo
    cfg.MODEL.WEIGHTS = "./output/model_1500.pth"
    # cfg.MODEL.WEIGHTS = "./output/model_final.pth"  # initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 1
    # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    cfg.SOLVER.BASE_LR = 0.0025
    # 300 iterations seems good enough, but you can certainly train longer
    cfg.SOLVER.MAX_ITER = (50)
    # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (256)
    cfg.OUTPUT_DIR = './'
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # 3 classes (data, fig, hazelnut)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)

    evaluator = COCOEvaluator(
        test_dataset, ("bbox", "segm"), False, output_dir="./output/")
    val_loader = build_detection_test_loader(cfg, test_dataset, num_workers=1)
    print(inference_on_dataset(trainer.model, val_loader, evaluator))
