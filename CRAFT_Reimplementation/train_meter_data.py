import os
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import scipy.io as scio
import argparse
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import random
# import h5py
import re
# import water
import json
from CRAFT_Reimplementation.data_loader import ICDAR2015, Synth80k, ICDAR2013, Meter_Dataset
import visdom
import xlrd
import xlwt
from CRAFT_Reimplementation.test_process import test
from CRAFT_Reimplementation.image.draw_polygon_result import draw_polygon_result
from math import exp

###import file#######
# from augmentation import random_rot, crop_img_bboxes
# from gaussianmap import gaussion_transform, four_point_transform
# from generateheatmap import add_character, generate_target, add_affinity, generate_affinity, sort_box, real_affinity, \
#     generate_affinity_box
from CRAFT_Reimplementation.mseloss import Maploss

from collections import OrderedDict
from CRAFT_Reimplementation.eval.script import getresult

from PIL import Image
from torchvision.transforms import transforms
from CRAFT_Reimplementation.craft import CRAFT
from torch.autograd import Variable
from multiprocessing import Pool

# 3.2768e-5
random.seed(42)

# class SynAnnotationTransform(object):
#     def __init__(self):
#         pass
#     def __call__(self, gt):
#         image_name = gt['imnames'][0]
parser = argparse.ArgumentParser(description='CRAFT reimplementation')

parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--batch_size', default=128, type=int,
                    help='batch size of training')
parser.add_argument('--cuda', default=True, type=bool,
help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate', default=3.2768e-5, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--num_workers', default=32, type=int,
                    help='Number of workers used in dataloading')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

vis = visdom.Visdom()

logger = "log_1218.txt"
with open(logger, 'r+') as file:
    file.truncate(0)
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (0.8 ** step)
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':

    # gaussian = gaussion_transform()
    # box = scio.loadmat('/data2/sunjunjiao/CRAFT_Reimplementation/SynthText/gt.mat')
    # bbox = box['wordBB'][0][0][0]
    # charbox = box['charBB'][0]
    # imgname = box['imnames'][0]
    # imgtxt = box['txt'][0]
    #
    # dataloader = syndata(imgname, charbox, imgtxt)
    # dataloader = Synth80k('/data2/sunjunjiao/CRAFT_Reimplementation/SynthText', target_size=768)
    # train_loader = torch.utils.data.DataLoader(
    #     dataloader,
    #     batch_size=2,
    #     shuffle=True,
    #     num_workers=0,
    #     drop_last=True,
    #     pin_memory=True)
    # batch_syn = iter(train_loader)
    # prefetcher = data_prefetcher(dataloader)
    # input, target1, target2 = prefetcher.next()
    # print(input.size())
    net = CRAFT()
    # net.load_state_dict(copyStateDict(torch.load('/data2/sunjunjiao/CRAFT_Reimplementation/CRAFT_net_050000.pth')))
    net.load_state_dict(copyStateDict(torch.load('/data2/sunjunjiao/CRAFT_Reimplementation/Syndata.pth')))
    # net.load_state_dict(copyStateDict(torch.load('/data2/sunjunjiao/CRAFT_Reimplementation/craft_mlt_25k.pth')))
    # net.load_state_dict(copyStateDict(torch.load('/data2/sunjunjiao/CRAFT_Reimplementation/synweights/syn_0_20000.pth')))
    # realdata = realdata(net)

    net = net.cuda()
    # net = CRAFT_net

    # if args.cdua:
    net = torch.nn.DataParallel(net, device_ids=[0]).cuda()
    # net.train()

    cudnn.benchmark = False
    realdata = Meter_Dataset(net, '/data2/sunjunjiao/data/pointer_meter1210', target_size=300, viz=False)
    real_data_loader = torch.utils.data.DataLoader(
        realdata,
        batch_size=10,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        pin_memory=True)

    # net.train()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = Maploss()
    # criterion = torch.nn.MSELoss(reduce=True, size_average=True)
    net.train()

    step_index = 0

    loss_time = 0
    loss_value = 0
    compare_loss = 1

    row_xls = 0
    writebook = xlwt.Workbook()  # 打开一个excel
    sheet = writebook.add_sheet('log_craft_12_18')  # 在打开的excel中添加一个sheet
    sheet.write(row_xls, 0, 'Model')  # 写入excel，  model
    sheet.write(row_xls, 1, 'Precition')  # 写入excel，  Precition
    sheet.write(row_xls, 2, 'Recall')  # 写入excel，  Recall
    sheet.write(row_xls, 3, 'H_mean')  # 写入excel，  H_mean
    sheet.write(row_xls, 4, 'Loss')  # 写入excel，  Loss
    writebook.save('log_1218.xls')  # 一定要记得保存
    row_xls+=1

    tplt = "{0:<15}\t{1:<35}\t{2:<35}\t{3:<35}\t{4:<35}"
    index_step = 100
    for epoch in range(1000):
        train_time_st = time.time()
        loss_value = 0
        if epoch % 27 == 0 and epoch != 0:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        st = time.time()
        epoch_mean_loss = []
        for index, (real_images, real_gh_label, real_gah_label, real_mask, _) in enumerate(real_data_loader):
            # net.train()
            # real_images, real_gh_label, real_gah_label, real_mask = next(batch_real)
            # syn_images, syn_gh_label, syn_gah_label, syn_mask, __ = next(batch_syn)
            # net.train()
            # images = torch.cat((syn_images, real_images), 0)
            # gh_label = torch.cat((syn_gh_label, real_gh_label), 0)
            # gah_label = torch.cat((syn_gah_label, real_gah_label), 0)
            # mask = torch.cat((syn_mask, real_mask), 0)

            images = real_images
            gh_label = real_gh_label
            gah_label = real_gah_label
            mask = real_mask
            # affinity_mask = torch.cat((syn_mask, real_affinity_mask), 0)

            images = Variable(images.type(torch.FloatTensor)).cuda()
            gh_label = gh_label.type(torch.FloatTensor)
            gah_label = gah_label.type(torch.FloatTensor)
            gh_label = Variable(gh_label).cuda()
            gah_label = Variable(gah_label).cuda()
            mask = mask.type(torch.FloatTensor)
            mask = Variable(mask).cuda()
            # affinity_mask = affinity_mask.type(torch.FloatTensor)
            # affinity_mask = Variable(affinity_mask).cuda()

            out, _ = net(images)

            optimizer.zero_grad()

            out1 = out[:, :, :, 0].cuda()
            out2 = out[:, :, :, 1].cuda()
            loss = criterion(gh_label, gah_label, out1, out2, mask)

            loss.backward()
            optimizer.step()
            loss_value += loss.item()
            if index % 10 == 0 and index > 0:
                et = time.time()
                print(
                    'epoch {}:({}/{}) batch || training time for 10 batch {} || training loss {} ||'.format(epoch, index,
                                                                                                           len(
                                                                                                               real_data_loader),
                                                                                                           et - st,
                                                                                                           loss_value / 10))
                epoch_mean_loss.append(loss_value / 10)
                loss_time = 0
                loss_value = 0
                st = time.time()
            # if loss < compare_loss:
            #     print('save the lower loss iter, loss:',loss)
            #     compare_loss = loss
            #     torch.save(net.module.state_dict(),
            #                '/data2/sunjunjiao/CRAFT_Reimplementation/real_weights/lower_loss.pth')

            # net.eval()
            if index % index_step == 0 and index != 0:

                print('Saving state, iter:', index)
                torch.save(net.module.state_dict(),
                           '/data2/sunjunjiao/CRAFT_Reimplementation/weights/mlt' + '_' + repr(epoch) + '_' + repr(index) + '.pth')
                test('/data2/sunjunjiao/CRAFT_Reimplementation/weights/mlt' + '_' + repr(epoch) + '_' + repr(index) + '.pth')
                # test('/data2/sunjunjiao/CRAFT_Reimplementation/craft_mlt_25k.pth')
                resDict = getresult()
                print('draw_polygon image from /result:')
                draw_polygon_result()
                # print(json.dumps(resDict['method']))

                i = json.dumps(resDict['method'])
                list = i.split(',')
                non_decimal = re.compile(r'[^\d.]+')
                precision = float(non_decimal.sub('', list[0]))
                recall = float(non_decimal.sub('', list[1]))
                hmean = float(non_decimal.sub('', list[2]))
                epoch_mean_loss_val = sum(epoch_mean_loss) / len(epoch_mean_loss)

                with open(logger, "a") as file:
                    file.write(tplt.format('mlt' + '_' + repr(epoch) + '_' + repr(index),
                                               'precition: ' + str(precision),
                                           'recall: ' + str(recall),
                                           'hmean: ' + str(hmean),
                                           'loss: ' + str(sum(epoch_mean_loss)/len(epoch_mean_loss))) + '\n')
                    # file.write('mlt' + '_' + repr(epoch) + '_' + repr(index) + '\t' + '\t' + 'precition: ' + str(precision) + '\t' + '\t' + 'recall: ' + str(recall) + '\t' + '\t' + 'hmean: ' + str(hmean) + '\n')

                sheet.write(row_xls, 0, 'mlt' + '_' + repr(epoch) + '_' + repr(index))  # 写入excel，  model
                sheet.write(row_xls, 1, precision)  # 写入excel，  precition
                sheet.write(row_xls, 2, recall)  # 写入excel，  recall
                sheet.write(row_xls, 3, hmean)  # 写入excel，  hmean
                sheet.write(row_xls, 4, epoch_mean_loss_val)  # 写入excel，  epoch_mean_loss_val
                writebook.save('log_1218.xls')  # 一定要记得保存
                row_xls+=1

        print('Saving state, iter:', epoch)
        torch.save(net.module.state_dict(),
                   '/data2/sunjunjiao/CRAFT_Reimplementation/epoch_weights/mlt' + '_' + repr(epoch) + '.pth')
        test('/data2/sunjunjiao/CRAFT_Reimplementation/epoch_weights/mlt' + '_' + repr(epoch) + '.pth')
        # test('/data2/sunjunjiao/CRAFT_Reimplementation/craft_mlt_25k.pth')
        resDict = getresult()
        print('draw_polygon image from /result:')
        draw_polygon_result()

        i = json.dumps(resDict['method'])
        list = i.split(',')
        non_decimal = re.compile(r'[^\d.]+')
        precision = float(non_decimal.sub('', list[0]))
        recall = float(non_decimal.sub('', list[1]))
        hmean = float(non_decimal.sub('', list[2]))

        with open(logger, "a") as file:
            file.write(tplt.format('mlt' + '_' + repr(epoch),
                                   'precition: ' + str(precision),
                                   'recall: ' + str(recall),
                                   'hmean: ' + str(hmean),
                                   'loss: ' + str(sum(epoch_mean_loss) / len(epoch_mean_loss))) + '\n')
            # file.write('mlt' + '_' + repr(epoch) + '----' + json.dumps(resDict['method']) + '\n')
            # file.write('mlt' + '_' + repr(epoch)  + '\t' + '\t' + 'precition: ' + str(precision) + '\t' + '\t' + 'recall: ' + str(recall) + '\t' + '\t' + 'hmean: ' + str(hmean) + '\n')


        # write to visdom
        x = torch.Tensor([epoch])
        epoch_mean_loss_val = sum(epoch_mean_loss)/len(epoch_mean_loss)
        vis.line(X=x, Y=torch.Tensor([epoch_mean_loss_val]), win='train loss', update='append', opts={'title': 'train_loss'})
        vis.line(X=x, Y=torch.Tensor([precision]), win='precision', update='append', opts={'title': 'precision'})
        vis.line(X=x, Y=torch.Tensor([recall]), win='recall', update='append', opts={'title': 'recall'})
        vis.line(X=x, Y=torch.Tensor([hmean]), win='hmean', update='append', opts={'title': 'hmean'})
        # vis.line(X=x, Y=torch.Tensor([train_te_iou]), win='train iou', update='append', opts={'title': 'train_iou'})
        # vis.line(X=x, Y=torch.Tensor([optimizer.param_groups[0]['lr']]), win='learning rate', update='append',
        #          opts={'title': 'learning_rate'})


        # write to excel
        # readbook = xlrd.open_workbook(r'log.xlsx')
        # sheet = readbook.sheet_by_name('log_craft_11_13')
        # nrows = sheet.nrows  # 行
        # ncols = sheet.ncols  # 列
        sheet.write(row_xls, 0, 'mlt' + '_' + repr(epoch))  # 写入excel，  model
        sheet.write(row_xls, 1, precision)  # 写入excel，  precition
        sheet.write(row_xls, 2, recall)  # 写入excel，  recall
        sheet.write(row_xls, 3, hmean)  # 写入excel，  hmean
        sheet.write(row_xls, 4, epoch_mean_loss_val)  # 写入excel，  epoch_mean_loss_val
        writebook.save('log_1218.xls')  # 一定要记得保存
        row_xls+=1
