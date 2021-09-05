from CRAFT_Reimplementation.test_process import test_net
from CRAFT_Reimplementation.test_process import copyStateDict

import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image
from CRAFT_Reimplementation.eval.script import getresult
import cv2
from skimage import io
import numpy as np
from CRAFT_Reimplementation import craft_utils
from CRAFT_Reimplementation import imgproc
from CRAFT_Reimplementation import file_utils
import json
import zipfile
from CRAFT_Reimplementation.image.draw_polygon_result import draw_polygon_result
from CRAFT_Reimplementation.craft import CRAFT
import re
import xlrd
import xlwt
from collections import OrderedDict

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.3, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.3, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.3, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda to train model')
parser.add_argument('--canvas_size', default=2240, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=2, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
args = parser.parse_args()
result_folder = 'demo_data_out'


def demo_for_meter_craft(model_craft, image_path):
    # load net
    net = CRAFT()  # initialize
    value_poly = []
    print('Loading weights from checkpoint {}'.format(model_craft))
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(model_craft)))
    else:
        net.load_state_dict(copyStateDict(torch.load(model_craft, map_location='cpu')))

    if args.cuda:
        # device = torch.device("cuda:0")
        net = net.cuda()
        print('cuda number', torch.cuda.device_count())
        net = torch.nn.DataParallel(net, device_ids=[0])

        # net.cuda()
        cudnn.benchmark = False

    net.eval()

    t = time.time()

    # load data
    print('test image :' + image_path + '\n')
    image = imgproc.loadImage(image_path)
    try:
        bboxes, polys, score_text = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text,
                                             args.cuda, args.poly)
    except:
        print("cuda error: ", image_path)

    # save score text
    filename, file_ext = os.path.splitext(os.path.basename(image_path))
    mask_file = result_folder + "/res_" + filename + '_mask.jpg'
    # cv2.imwrite(mask_file, score_text)
    # file_utils.saveResult(image_path, image[:, :, ::-1], polys, dirname=result_folder)

    for i, box in enumerate(polys):
        poly = np.array(box).astype(np.int32).reshape((-1))
        strResult = ','.join([str(p) for p in poly])
        value_poly.append(strResult)
    return value_poly


if __name__ == '__main__':
    value = demo_for_meter_craft('epoch_weights/mlt_19.pth',
                                 'demo_data/pointer_15.png')
    print(value)