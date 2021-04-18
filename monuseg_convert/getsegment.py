import xml.dom.minidom
import numpy as np
from cv2 import cv2
from skimage import draw


def polygons_to_mask(img_shape, polygons):
    mask = np.zeros(img_shape, dtype=np.uint8)
    polygons = np.asarray(polygons, np.int32)  # 这里必须是int32，其他类型使用fillPoly会报错
    shape = polygons.shape
    polygons = polygons.reshape(shape[0], -1, 2)
    cv2.fillPoly(mask, polygons, color=1)  # 非int32 会报错

    return mask


def mask2polygon(mask):
    contours, hierarchy = cv2.findContours(
        (mask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # mask_new, contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for contour in contours:
        contour_list = contour.flatten().tolist()
        if len(contour_list) > 4:  # and cv2.contourArea(contour)>10000
            segmentation.append(contour_list)
    return segmentation


def poly2mask(vertex_row_coords, vertex_col_coords, shape):
    fill_row_coords, fill_col_coords = draw.polygon(
        vertex_row_coords, vertex_col_coords, shape)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[fill_row_coords, fill_col_coords] = 255
    return mask


def get_segmentation(file_name):
    original_file_name = file_name.split(".")[0]
    xDoc = xml.dom.minidom.parse(original_file_name+'.xml')
    Regions = xDoc.getElementsByTagName('Region')
    xy = {}
    for regioni in range(0, Regions.length):
        Region = Regions.item(regioni)

        verticies = Region.getElementsByTagName('Vertex')
        xy[regioni] = np.zeros((verticies.length, 2))
        for vertexi in range(0, verticies.length):
            x = float(verticies.item(vertexi).getAttribute('X'))

            y = float(verticies.item(vertexi).getAttribute('Y'))
            xy[regioni][vertexi, :] = [x, y]
    print(len(xy))
    image = cv2.imread(file_name)
    mask_list = []
    for zz in range(0, len(xy)):
        print('creating masks #', zz, '\n')
        smaller_x = xy[zz][:, 0]
        smaller_y = xy[zz][:, 1]
        mask = poly2mask(smaller_x, smaller_y,
                         (image.shape[0], image.shape[1]))
        mask_list.append(mask)
    return mask_list
