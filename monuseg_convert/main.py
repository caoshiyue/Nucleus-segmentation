
import cv2
import numpy as np
from matplotlib import pyplot as plt
from getsegment import *
from converter import *
from PIL import Image
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon
import os
import json
import glob

img_suffix = '*.tif'
category_ids = {
    "cell": 1,
}


def gen_coco_anno(path):
    coco_format = get_coco_json_format()
    coco_format["categories"] = create_category_annotation(category_ids)
    image_id = 0
    annotation_id = 0
    coco_format["images"] = []
    coco_format["annotations"] = []
    for f in glob.glob(os.path.join(path, img_suffix)):
        print(f)
        mask_list = get_segmentation(f)
        images, annotations, image_id, annotation_id = images_annotations_info(
            mask_list, f, image_id, annotation_id)

        coco_format["images"] = coco_format["images"]+images
        coco_format["annotations"] = coco_format["annotations"] + annotations
    return coco_format


if __name__ == "__main__":
    for fold in ["train", "test"]:  # no ./ in path
        coco_format = gen_coco_anno('data/'+fold)
        with open('data/'+fold+"/{}.json".format(fold), "w") as outfile:
            json.dump(coco_format, outfile)
