import random
from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
import data_register
import cv2
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
import os
setup_logger()


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
    cfg.MODEL.WEIGHTS = "model_final_f10217.pkl"
    # cfg.MODEL.WEIGHTS = "./output/model_final.pth"  # initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 0.0025
    # 300 iterations seems good enough, but you can certainly train longer
    cfg.SOLVER.MAX_ITER = (50)
    # faster, and good enough for this toy dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (256)
    cfg.OUTPUT_DIR = './'
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
