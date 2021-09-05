import os

import numpy as np
import cv2

from E2E_MLT import net_utils
import torch
import unicodedata as ud
from E2E_MLT.models import ModelResNetSep2
from Levenshtein import distance
import pandas as pd
import argparse
from E2E_MLT.ocr_utils import print_seq_ext
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw


def demo_for_meter_e2e_path(model_e2e, codec_path, image_path, norm_height=32):
    f = open(codec_path, 'r', encoding='utf-8')
    codec = f.readlines()[0]
    f.close()

    parser = argparse.ArgumentParser()
    parser.add_argument('-cuda', type=int, default=1)
    parser.add_argument('-segm_thresh', default=0.5)

    font2 = ImageFont.truetype("Arial-Unicode-Regular.ttf", 18)

    args = parser.parse_args()

    net = ModelResNetSep2(attention=True)
    net_utils.load_net(model_e2e, net)
    net = net.eval()

    if args.cuda:
        print('Using cuda ...')
        net = net.cuda()

    scripts = ['', 'DIGIT', 'LATIN', 'ARABIC', 'BENGALI', 'HANGUL', 'CJK', 'HIRAGANA', 'KATAKANA']

    conf_matrix = np.zeros((len(scripts), len(scripts)), dtype=np.int)

    gt_script = {}
    ed_script = {}
    correct_ed1_script = {}
    correct_script = {}
    count_script = {}
    for scr in scripts:
        gt_script[scr] = 0
        ed_script[scr] = 0
        correct_script[scr] = 0
        correct_ed1_script[scr] = 0
        count_script[scr] = 0

    it = 0
    it2 = 0
    correct = 0
    correct_ed1 = 0
    ted = 0
    gt_all = 0
    images_count = 0
    bad_words = []

    image_name = image_path

    img = cv2.imread(image_name)
    scale = norm_height / float(img.shape[0])
    width = int(img.shape[1] * scale)
    width = max(8, int(round(width / 4)) * 4)

    scaled = cv2.resize(img, (int(width), norm_height))
    # scaled = scaled[:, :, ::-1]
    scaled = np.expand_dims(scaled, axis=0)

    scaled = np.asarray(scaled, dtype=np.float)
    scaled /= 128
    scaled -= 1

    try:
        scaled_var = net_utils.np_to_variable(scaled, is_cuda=args.cuda).permute(0, 3, 1, 2)
        x = net.forward_features(scaled_var)
        ctc_f = net.forward_ocr(x)
        ctc_f = ctc_f.data.cpu().numpy()
        ctc_f = ctc_f.swapaxes(1, 2)

        labels = ctc_f.argmax(2)
        det_text, conf, dec_s, _ = print_seq_ext(labels[0, :], codec)
    except:
        print('bad image')
        det_text = ''
    det_text = det_text.strip()

    return det_text


def demo_for_meter_e2e(model_e2e, codec_path, crop_img, norm_height=32):
    f = open(codec_path, 'r', encoding='utf-8')
    codec = f.readlines()[0]
    f.close()

    parser = argparse.ArgumentParser()
    parser.add_argument('-cuda', type=int, default=1)
    parser.add_argument('-segm_thresh', default=0.5)

    font2 = ImageFont.truetype("E2E_MLT/Arial-Unicode-Regular.ttf", 18)

    args = parser.parse_args()

    net = ModelResNetSep2(attention=True)
    net_utils.load_net(model_e2e, net)
    net = net.eval()

    if args.cuda:
        # print('Using cuda ...')
        net = net.cuda()

    scripts = ['', 'DIGIT', 'LATIN', 'ARABIC', 'BENGALI', 'HANGUL', 'CJK', 'HIRAGANA', 'KATAKANA']

    conf_matrix = np.zeros((len(scripts), len(scripts)), dtype=np.int)

    gt_script = {}
    ed_script = {}
    correct_ed1_script = {}
    correct_script = {}
    count_script = {}
    for scr in scripts:
        gt_script[scr] = 0
        ed_script[scr] = 0
        correct_script[scr] = 0
        correct_ed1_script[scr] = 0
        count_script[scr] = 0

    img = crop_img
    try:
        scale = norm_height / float(img.shape[0])
        width = int(img.shape[1] * scale)
        width = max(8, int(round(width / 4)) * 4)

        scaled = cv2.resize(img, (int(width), norm_height))
        # scaled = scaled[:, :, ::-1]
        scaled = np.expand_dims(scaled, axis=0)

        scaled = np.asarray(scaled, dtype=np.float)
        scaled /= 128
        scaled -= 1

        scaled_var = net_utils.np_to_variable(scaled, is_cuda=args.cuda).permute(0, 3, 1, 2)
        x = net.forward_features(scaled_var)
        ctc_f = net.forward_ocr(x)
        ctc_f = ctc_f.data.cpu().numpy()
        ctc_f = ctc_f.swapaxes(1, 2)

        labels = ctc_f.argmax(2)
        det_text, conf, dec_s, _ = print_seq_ext(labels[0, :], codec)
    except:
        print('bad image')
        det_text = ''
    det_text = det_text.strip()

    return det_text


if __name__ == '__main__':
    value = demo_for_meter_e2e_path('backup_1222/E2E_11000.h5',
                                    'codec_1211.txt',
                                    'demo_data/0_Image_2020061517385373_9.jpg'
                                    )
    print(value)
