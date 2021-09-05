from __future__ import division
from __future__ import print_function
# import onnx
# import os
import argparse
import cv2
# import onnxruntime
import time
from threading import Thread, enumerate
import sys

if sys.version > '3':
    import queue as Queue
else:
    import Queue

from scipy import optimize
import functools
import math
# from statistics import *
import numpy as np
from numpy import *
from scipy import optimize
import functools
from CRAFT_Reimplementation import file_utils
import math
from itertools import combinations, permutations
from meter_ocr_api import meter_ocr_api

# from tool.utils import *
# from tool.torch_utils import *
sigma_list = [15, 80, 250]
low_clip = 0.01
high_clip = 0.99
Meter_IS_COMPLETE_LEFT_RIGHT = 200
Meter_IS_COMPLETE_TOP_DOWN = 5
Meter_SCORE_SIZE = 0
Meter_AREA_RATIO_MIN = 0.05
Meter_AREA_RATIO_MAX = 0.5
Meter_ARTICUL_SIZE = 0.05
ERROR_Meter_VALUE_ARTICUL_SIZE = -102  # 清晰度
ERROR_Meter_VALUE_SCORE_SIZE = -103
ERROR_Meter_VALUE_AREA_RATIO_MIN = -104  # 最小面积
ERROR_Meter_VALUE_AREA_RATIO_MAX = -105  # 最大面积
ERROR_Meter_VALUE_ASPECT_RATIO_MIN = -106
ERROR_Meter_VALUE_ASPECT_RATIO_MAX = -107
ERROR_Meter_VALUE_IS_COMPLETE_LEFT_RIGHT = -108
ERROR_Meter_VALUE_IS_COMPLETE_TOP_DOWN = -109
ERROR_Meter_NO_FOUND_POINTER = -110
ERROR_Meter_FabsValue = -111

_Not_Available_Pic = 111


class my_dictionary(dict):

    def __init__(self):
        self = dict()

    def add(self, key, value):
        self[key] = value


class GetCenter(object):
    def __init__(self, image, x_cor, y_cor):
        self.x_list = x_cor
        self.y_list = y_cor
        self.image = image
        self.x_m = mean(self.x_list)
        self.y_m = mean(self.y_list)
        center_estimate = self.x_m, self.y_m
        center_2, _ = optimize.leastsq(f_2, center_estimate)
        self.center = center_2

    def return_center(self):
        return self.center


def countcalls(fn):
    "decorator function count function calls "

    @functools.wraps(fn)
    def wrapped(*args):
        wrapped.ncalls += 1
        return fn(*args)

    wrapped.ncalls = 0
    return wrapped


@countcalls
def f_2(c):
    Ri = calc_R(*c)
    return Ri - Ri.mean()


def calc_R(xc, yc):
    return sqrt((result_x - xc) ** 2 + (result_y - yc) ** 2)


class TrasferPolar(object):
    def __init__(self, image, input_center):
        self.image = image
        self.center = input_center
        print("-----estimate circle center----", self.center)
        # self.after_transfer, x, y, half_diagonal = self.polar(self.image, self.center, (0,340))
        # polar_x,polar_y = self.polar_coordinate(x,y,half_diagonal)
        # duiying_x, duiying_y = self.get_original_coordinate(self.image, self.center, (0,340), polar_x, polar_y)
        # self.original_x = duiying_x
        # self.original_y = duiying_y

    def polar(self, r, ocr_y=None, ocr_x=None, theta=(0, 360), rstep=0.5, thetastep=360.0 / (180 * 4)):
        # 得到距离的最小值、最大值
        txt_coordinate_original = []
        for xx in range(len(ocr_x)):
            txt_coordinate_original.append([ocr_y[xx], ocr_x[xx]])
        image = self.image
        center = self.center

        minr, maxr = r
        cx, cy = center
        # print("minr, maxr: ", minr, maxr)
        # 角度的最小范围
        mintheta, maxtheta = theta
        # 输出图像的高、宽 O:指定形状类型的数组float64
        H = int((maxr - minr) / rstep) + 1
        W = int((maxtheta - mintheta) / thetastep) + 1

        # new image size H, W
        print("polar_image H:", H)
        print("polar_image W:", W)
        h, w = image.shape[:2]

        O = 125 * np.ones((H, W), image.dtype)
        # 极坐标转换  利用tile函数实现W*1铺成的r个矩阵 并对生成的矩阵进行转置
        r = np.linspace(minr, maxr, H)
        r = np.tile(r, (W, 1))
        r = np.transpose(r)
        theta = np.linspace(mintheta, maxtheta, W)
        theta = np.tile(theta, (H, 1))
        # 极坐标转为笛卡尔坐标，x, y 的宽高为变换后的坐标系
        x, y = cv2.polarToCart(r, theta, angleInDegrees=True)

        # 505，525 为原始中心点的坐标
        # 最近插值法
        # py图像的高, px图像的宽，对应的原始坐标下的数值。

        flag = 0
        for i in range(H):
            for j in range(W):

                # 88.52126693725586 508.5004577636719
                # tmp_i =
                px = int(round(x[i][j]) + cx)
                py = int(round(y[i][j]) + cy)

                if ((px >= 0 and px <= w - 1) and (py >= 0 and py <= h - 1)):
                    O[i][j] = image[py][px]
                # 08 文字中心点坐标
                # if(py == (526)) and (px == (505)):
                # print("中心点坐标j,i", j, i)
                # for zz in range(len(txt_coordinate_original)):
                #     if py == txt_coordinate_original[zz][1] and px == txt_coordinate_original[zz][0]:
                #         #flag = 1
                #         final_x.append(j)
                #         final_y.append(i)

                # if py == txt_coordinate_original and px == ocr_y :
                #     flag = 1
                #     final_x = j
                #     final_y = i
                # elif (py == (ocr_x)) and ( (ocr_y - 3)<= px <=(ocr_y + 3)) and (px != ocr_y):
                #     #print(round(x[i][j]))
                #     final_x = j
                #     final_y = i

                # print("中心点坐标j,i", j, i)
                # if(py == (546)) and (px == (478)):
                #     print("左下角坐标j,i", j, i)
                # if(j == 88) and(i == 508):
                #     print(px, py)
        # for i in range(H):
        #     for j in range(W):

        #         px = int(round(x[i][j]) + cx)
        #         py = int(round(y[i][j]) + cy)

        #         # 08 文字中心点坐标
        #         #if(py == (526)) and (px == (505)):

        #             #print("中心点坐标j,i", j, i)
        #         if ((ocr_x - 5) <= py <= (ocr_x + 5)) and ( (ocr_y - 5)<= px <=(ocr_y + 5)) and flag == 0 :
        #             final_x = j
        #             final_y = i

        # print("polar center and half_diagonal", final_x, final_y, half_diagonal)
        return O, x, y, H, W

    def polarcor(self, x_, y_, H, W, ocr_y, ocr_x):

        x = x_
        y = y_
        cx, cy = self.center
        # print("minr, maxr: ", minr, maxr)
        # 角度的最小范围
        flag = 0
        # print("传过来的H,W", H, W)

        for i in range(H):
            for j in range(W):
                px = int(round(x[i][j]) + cx)
                py = int(round(y[i][j]) + cy)
                if py == ocr_x and px == ocr_y:
                    # print("------------进入了---------------")
                    flag = 1
                    final_x = j
                    final_y = i
                    continue

        if flag == 0:
            for i in range(H):
                for j in range(W):
                    px = int(round(x[i][j]) + cx)
                    py = int(round(y[i][j]) + cy)
                    if ((ocr_x - 5) <= py <= (ocr_x + 5)) and ((ocr_y - 5) <= px <= (ocr_y + 5)):
                        final_x = j
                        final_y = i

        return final_x, final_y


if __name__ == '__main__':
    image_list, _, _ = file_utils.get_files('demo_data_polar')
    for image_path in image_list:
        try:
            print(image_path)
            results = meter_ocr_api(image_path)
            result_x = []
            result_y = []
            result_h = []
            result_w = []
            for result in results:
                result_x.append(result.x + int(result.w) / 2)
                result_y.append(result.y + int(result.h) / 2)
                result_h.append(result.h)
                result_w.append(result.w)

            img_for_keduxian = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            bb = GetCenter(img_for_keduxian, result_x, result_y)
            tmp_center = bb.return_center()

            get_polar_pic = TrasferPolar(img_for_keduxian, tmp_center)

            final_polar_image, tmp_x, tmp_y, tmp_H, tmp_W = get_polar_pic.polar((0, 340),
                                                                                result_x,
                                                                                result_y)
            image_out = image_path.replace('demo_data_polar', 'demo_data_polar_out')
            cv2.imwrite(image_out.replace('.png', '_out.png'), final_polar_image)
        except:
            print('error in ' + image_path)
