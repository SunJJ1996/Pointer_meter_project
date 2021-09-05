import os

from filter_result_number import filter_if_has_number
from filter_result_number import filter_height_Zscore
from filter_result_number import filter_distence_Zscore
from filter_result_number import filter_area_Zscore
from filter_result_number import filter_convex_hull
from filter_result_number import filter_edge
from filter_result_number import filter_height_mean
import cv2
from CRAFT_Reimplementation.demo_for_meter_craft import demo_for_meter_craft
from E2E_MLT.demo_for_meter_e2e import demo_for_meter_e2e
from E2E_MLT.images.crop_for_ocr import crop_image
import numpy as np
from CRAFT_Reimplementation.image.draw_polygon_result import draw_polygon_and_word


class meter_ocr:
    def __init__(self):
        self.content = ''
        self.bbox_list = []  # 列表
        self.unit = ''
        self.manufacturer = ''
        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0


def count_core(image):
    img = cv2.imread(image)
    h = img.shape[0]
    w = img.shape[1]
    x = w / 2
    y = h / 2
    return x, y


def demo_meter_text(img_path, craft_weight, e2e_weight, codec_path):
    results = []
    values = demo_for_meter_craft(craft_weight,
                                  img_path)
    for value in values:
        result = meter_ocr()
        # print(value)
        bboxtemp = value.split(',')
        bbox_np = np.array(bboxtemp, dtype=np.int32)
        bbox = bbox_np.reshape(4, 2)
        # 直边外接矩形
        x, y, w, h = cv2.boundingRect(bbox)
        img = cv2.imread(img_path)
        crop_img = crop_image(img, x, y, x + w, y + h)
        content = demo_for_meter_e2e(e2e_weight,
                                     codec_path,
                                     crop_img)
        # print(content)
        result.bbox_list = bbox
        result.content = content
        result.x = x
        result.y = y
        result.h = h
        result.w = w
        results.append(result)
    # print(results[0].bbox_list)
    print('text recognition and ocr has done')
    return results


def meter_ocr_api(img_path):
    craft_w = 'CRAFT_Reimplementation/weights/mlt_3_1100.pth'
    e2e_w = 'E2E_MLT/backup_1222/E2E_11000.h5'
    codec_p = 'E2E_MLT/codec_1211.txt'

    img = cv2.imread(img_path)
    img_x, img_y = img.shape[0:2]
    scale_x = img_x / 300
    scale_y = img_y / 300
    img = cv2.resize(img, (300, 300), interpolation=cv2.INTER_CUBIC)
    resize_path = os.path.splitext(img_path)[0]
    resize_sp = os.path.splitext(img_path)[-1]
    new_path = resize_path + '_resize' + resize_sp
    cv2.imwrite(new_path, img)
    img_path = new_path

    results_origin = demo_meter_text(img_path, craft_w, e2e_w, codec_p)
    # filter edge
    img_x, img_y = count_core(img_path)
    results_NoEdge = filter_edge(results_origin, img_x, img_y)

    # filter all not number result
    results_onlyNum = filter_if_has_number(results_NoEdge)

    # filter convex_hull
    results_convex_hull = filter_convex_hull(results_onlyNum)

    # filter height
    results_filter_height_mean = filter_height_mean(results_convex_hull)
    results_resize = results_filter_height_mean

    # resume to origin size

    for result_resize in results_resize:
        result_resize.bbox_list[0][0] *= scale_y
        result_resize.bbox_list[1][0] *= scale_y
        result_resize.bbox_list[2][0] *= scale_y
        result_resize.bbox_list[3][0] *= scale_y
        result_resize.bbox_list[0][1] *= scale_x
        result_resize.bbox_list[1][1] *= scale_x
        result_resize.bbox_list[2][1] *= scale_x
        result_resize.bbox_list[3][1] *= scale_x
        x, y, w, h = cv2.boundingRect(result_resize.bbox_list)
        result_resize.x = x
        result_resize.y = y
        result_resize.h = h
        result_resize.w = w
    os.remove(new_path)
    return results_resize
if __name__ == '__main__':
    image_path = 'demo_data_polar/pointer_23.png'
    results = meter_ocr_api(image_path)
    # draw_polygon_and_word(image_path, 'demo_data_out', results)
    for result in results:
        # print(result.bbox_list)
        print(result.x)
        print(result.y)
        print(result.h)
        print(result.w)
        print('content:' + result.content)


