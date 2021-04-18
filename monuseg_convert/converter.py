# (pip install Pillow)
from PIL import Image
# (pip install numpy)
import numpy as np
# (pip install scikit-image)
from skimage import measure
# (pip install Shapely)
from shapely.geometry import Polygon, MultiPolygon
import os
import json
from cv2 import cv2


def create_sub_masks(mask_image, width, height):
    # Initialize a dictionary of sub-masks indexed by RGB colors
    sub_masks = {}
    for x in range(width):
        for y in range(height):
            # Get the RGB values of the pixel
            pixel = mask_image.getpixel((x, y))[:3]

            # Check to see if we have created a sub-mask...
            pixel_str = str(pixel)
            sub_mask = sub_masks.get(pixel_str)
            if sub_mask is None:
               # Create a sub-mask (one bit per pixel) and add to the dictionary
                # Note: we add 1 pixel of padding in each direction
                # because the contours module doesn"t handle cases
                # where pixels bleed to the edge of the image
                sub_masks[pixel_str] = Image.new("1", (width+2, height+2))

            # Set the pixel value to 1 (default is 0), accounting for padding
            sub_masks[pixel_str].putpixel((x+1, y+1), 1)

    return sub_masks


def create_sub_mask_annotation(sub_mask):
    # Find contours (boundary lines) around each sub-mask
    # Note: there could be multiple contours if the object
    # is partially occluded. (E.g. an elephant behind a tree)

    # contours = measure.find_contours(
    #     np.array(sub_mask), 100, positive_orientation="low")
    contours, hierarchy = cv2.findContours(
        (sub_mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

    polygons = []
    segmentations = []

    for contour in contours:
        # Flip from (row, col) representation to (x, y)
        # and subtract the padding pixel
        new_contour = []
        if len(contour) < 4:
            continue
        for i in range(len(contour)):
            row, col = contour[i][0]
            #contour[i] = (col - 1, row - 1)
            if (col, row) in new_contour:  # multipolygon error
                continue
            new_contour.append((col, row))
        # Make a polygon and simplify it
        poly = Polygon(new_contour)
        poly = poly.simplify(1.0, preserve_topology=False)

        polygons.append(poly)

        segmentation = np.array(poly.exterior.coords).ravel().tolist()
        segmentations.append(segmentation)

    return polygons, segmentations


def create_category_annotation(category_dict):
    category_list = []

    for key, value in category_dict.items():
        category = {
            "supercategory": key,
            "id": value,
            "name": key
        }
        category_list.append(category)

    return category_list


def create_image_annotation(file_name, width, height, image_id):
    images = {
        "file_name": file_name,
        "height": height,
        "width": width,
        "id": image_id
    }

    return images


def create_annotation_format(polygon, segmentation, image_id, category_id, annotation_id):
    min_x, min_y, max_x, max_y = polygon.bounds
    width = max_x - min_x
    height = max_y - min_y
    bbox = (min_x, min_y, width, height)
    area = polygon.area

    annotation = {
        "segmentation": segmentation,
        "area": area,
        "iscrowd": 0,
        "image_id": image_id,
        "bbox": bbox,
        "category_id": category_id,
        "id": annotation_id
    }

    return annotation


def get_coco_json_format():
    # Standard COCO format
    coco_format = {
        "info": {},
        "licenses": [],
        "images": [{}],
        "categories": [{}],
        "annotations": [{}]
    }

    return coco_format


def images_annotations_info(seg_list, file_path, image_id, annotation_id):
    # file_path='train/1.tif'
    category_id = 1  # Todo: multi-category
    annotations = []
    images = []
    original_file_name = os.path.basename(
        file_path)

    original_image_open = cv2.imread(file_path)
    w, h, c = original_image_open.shape
    k = 0
    for seg in seg_list:
        print('Processing segmentation #', k, '\n')
        polygons, segmentations = create_sub_mask_annotation(seg)
        k = k+1
        for i in range(len(polygons)):
            # Cleaner to recalculate this variable
            segmentation = [
                np.array(polygons[i].exterior.coords).ravel().tolist()]

            annotation = create_annotation_format(
                polygons[i], segmentation, image_id, category_id, annotation_id)

            annotations.append(annotation)
            annotation_id += 1

    image = create_image_annotation(original_file_name, w, h, image_id)
    images.append(image)
    image_id += 1

    return images, annotations, image_id, annotation_id
