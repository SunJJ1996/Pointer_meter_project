from CRAFT_Reimplementation.demo_for_meter_craft import demo_for_meter_craft
from E2E_MLT.demo_for_meter_e2e import demo_for_meter_e2e
import cv2
from E2E_MLT.images.crop_for_ocr import crop_image
import numpy as np
from CRAFT_Reimplementation.image.draw_polygon_result import draw_polygon_and_word
from CRAFT_Reimplementation import file_utils
from random import sample
import os
import shutil
from filter_result_number import filter_if_has_number
from filter_result_number import filter_height_Zscore
from filter_result_number import filter_distence_Zscore
from filter_result_number import filter_area_Zscore
from filter_result_number import filter_convex_hull
from filter_result_number import filter_edge
from filter_result_number import filter_height_mean


class meter_ocr:
    def __init__(self):
        self.content = ''
        self.bbox_list = []  # 列表
        self.unit = ''
        self.manufacturer = ''


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
        results.append(result)
    # print(results[0].bbox_list)
    print('text recognition and ocr has done')
    return results


def joint_img(origin_path, filter_edge_path, filter_num_path, filter_convex_path, filter_h_mean_path, filter_h_zsore_path,out_path):
    image_list_origin, _, _ = file_utils.get_files(origin_path)
    image_list_edge, _, _ = file_utils.get_files(filter_edge_path)
    image_list_filter_num, _, _ = file_utils.get_files(filter_num_path)
    image_list_filter_convex, _, _ = file_utils.get_files(filter_convex_path)
    image_list_filter_h_mean, _, _ = file_utils.get_files(filter_h_mean_path)
    image_list_filter_h_zsore, _, _ = file_utils.get_files(filter_h_zsore_path)
    for img_origin in image_list_origin:
        filepath, fullflname_origin = os.path.split(img_origin)
        for img_filter_edge in image_list_edge:
            filepath, fullflname_filter_edge = os.path.split(img_filter_edge)
            for img_filter_num in image_list_filter_num:
                filepath, fullflname_filter_num = os.path.split(img_filter_num)
                for img_filter_convex in image_list_filter_convex:
                    filepath, fullflname_filter_convex = os.path.split(img_filter_convex)
                    for img_filter_h_mean in image_list_filter_h_mean:
                        filepath, fullflname_filter_h_mean = os.path.split(img_filter_h_mean)
                        for img_filter_h_zscore in image_list_filter_h_zsore:
                            filepath, fullflname_filter_h_zscore = os.path.split(img_filter_h_zscore)
                            if fullflname_origin == fullflname_filter_num and \
                                    fullflname_origin == fullflname_filter_convex and \
                                    fullflname_origin == fullflname_filter_h_mean and \
                                    fullflname_origin == fullflname_filter_edge and \
                                    fullflname_origin == fullflname_filter_h_zscore:
                                part1 = cv2.imread(img_origin)
                                part2 = cv2.imread(img_filter_edge)
                                part3 = cv2.imread(img_filter_num)
                                part4 = cv2.imread(img_filter_convex)
                                part5 = cv2.imread(img_filter_h_mean)
                                part6 = cv2.imread(img_filter_h_zscore)
                                res_img123 = np.hstack([part1, part2, part3])
                                res_img456 = np.hstack([part4, part5, part6])
                                res_img = np.vstack([res_img123, res_img456])
                                img_out_path = img_origin.replace(os.path.dirname(img_origin), out_path)
                                cv2.imwrite(img_out_path, res_img)


def count_core(image):
    img = cv2.imread(image)
    h = img.shape[0]
    w = img.shape[1]
    x = w / 2
    y = h / 2
    return x, y


if __name__ == '__main__':
    # img_path = 'demo_data/pointer_15.png'
    craft_w = 'CRAFT_Reimplementation/weights/mlt_3_1100.pth'
    e2e_w = 'E2E_MLT/backup_1222/E2E_11000.h5'
    codec_p = 'E2E_MLT/codec_1211.txt'
    # results = demo_meter_text(img_path, craft_w, e2e_w, codec_p)
    #
    # print('start draw in ' + img_path)
    # draw_polygon_and_word(img_path, out_path, results)

    image_list_origin, _, _ = file_utils.get_files('demo_data_new')
    # image_list, _, _ = file_utils.get_files('demo_data_pick')
    # image_list = sample(image_list_origin, 15)
    # image_list = image_list_origin[90:100]
    image_list = image_list_origin
    index = 0
    #
    out_path = 'demo_data_out'
    shutil.rmtree(out_path)
    os.mkdir(out_path)

    out_path_origin = 'demo_data_filter/demo_data_out_origin'
    shutil.rmtree(out_path_origin)
    os.mkdir(out_path_origin)

    out_path_filter_edge = 'demo_data_filter/demo_data_out_filter_edge'
    shutil.rmtree(out_path_filter_edge)
    os.mkdir(out_path_filter_edge)

    out_path_filter_num = 'demo_data_filter/demo_data_out_filter_num'
    shutil.rmtree(out_path_filter_num)
    os.mkdir(out_path_filter_num)

    out_path_filter_dis = 'demo_data_filter/demo_data_out_filter_dis'
    shutil.rmtree(out_path_filter_dis)
    os.mkdir(out_path_filter_dis)

    out_path_filter_area = 'demo_data_filter/demo_data_out_filter_area'
    shutil.rmtree(out_path_filter_area)
    os.mkdir(out_path_filter_area)

    out_path_filter_convex_hull = 'demo_data_filter/demo_data_out_convex_hull'
    shutil.rmtree(out_path_filter_convex_hull)
    os.mkdir(out_path_filter_convex_hull)

    out_path_filter_h_mean = 'demo_data_filter/demo_data_out_filter_h_mean'
    shutil.rmtree(out_path_filter_h_mean)
    os.mkdir(out_path_filter_h_mean)

    out_path_filter_h_zscore = 'demo_data_filter/demo_data_out_filter_h_zscore'
    shutil.rmtree(out_path_filter_h_zscore)
    os.mkdir(out_path_filter_h_zscore)

    for img in image_list:
        results = demo_meter_text(img, craft_w, e2e_w, codec_p)
        draw_polygon_and_word(img, out_path_origin, results)

        # filter edge
        img_x, img_y = count_core(img)
        results_NoEdge = filter_edge(results, img_x, img_y)
        draw_polygon_and_word(img, out_path_filter_edge, results_NoEdge)

        # filter all not number result
        results_onlyNum = filter_if_has_number(results_NoEdge)
        draw_polygon_and_word(img, out_path_filter_num, results_onlyNum)

        # filter distence from center
        # img_x, img_y = count_core(img)
        # results_filter_distence = filter_distence_Zscore(results_onlyNum, img_x, img_y)
        # draw_polygon_and_word(img, out_path_filter_dis, results_filter_distence)

        # filter area
        # results_filter_area = filter_area_Zscore(results_onlyNum)
        # draw_polygon_and_word(img, out_path_filter_area, results_filter_area)

        # filter convex_hull
        results_convex_hull = filter_convex_hull(results_onlyNum)
        draw_polygon_and_word(img, out_path_filter_convex_hull, results_convex_hull)

        # filter height
        results_filter_height_mean = filter_height_mean(results_convex_hull)
        draw_polygon_and_word(img, out_path_filter_h_mean, results_filter_height_mean)

        results_filter_height_zscore = filter_height_Zscore(results_convex_hull)
        draw_polygon_and_word(img, out_path_filter_h_zscore, results_filter_height_zscore)
        #
        joint_img(out_path_origin,
                  out_path_filter_edge,
                  out_path_filter_num,
                  out_path_filter_convex_hull,
                  out_path_filter_h_mean,
                  out_path_filter_h_zscore,
                  out_path)

        index += 1
        print(index)
