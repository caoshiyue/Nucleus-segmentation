import random
from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
import data_register
from cv2 import cv2


fruits_nuts_metadata = MetadataCatalog.get("test")
print(fruits_nuts_metadata)
dataset_dicts = DatasetCatalog.get("test")


d = dataset_dicts[1]
img = cv2.imread(d["file_name"])
visualizer = Visualizer(
    img[:, :, ::-1], metadata=fruits_nuts_metadata, scale=1)
vis = visualizer.draw_dataset_dict(d)

img = vis.get_image()[:, :, ::-1]
cv2.imshow('rr', img)
cv2.waitKey(0)
