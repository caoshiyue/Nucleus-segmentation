
from detectron2.data.datasets import register_coco_instances

register_coco_instances(
    "train", {}, "data/train/train.json", "data/train")
register_coco_instances(
    "test", {}, "data/test/test.json", "data/test")
