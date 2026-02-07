import os
import math
from datetime import datetime
import random
import logging
from collections import OrderedDict
import numpy as np
import cv2
import torch
import shutil

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

import torch.nn as nn
import torch.nn.functional as F


def OrderedYaml():
    '''yaml orderedDict support'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


####################
# miscellaneous
####################


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        logger = logging.getLogger('base')
        logger.info('Path already exists. Renove it.')
        shutil.rmtree(path)
        # new_name = path + '_archived_' + get_timestamp()
        # print('Path already exists. Rename it to [{:s}]'.format(new_name))
        # logger = logging.getLogger('base')
        # logger.info('Path already exists. Rename it to [{:s}]'.format(new_name))
        # os.rename(path, new_name)
    os.makedirs(path)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''Converts a torch Tensor into an image Numpy array'''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    img_np = tensor.numpy()
    img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
    elif out_type == np.uint16:
        img_np = (img_np * 65535.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, img)


def save_npy(img, img_path):
    img = np.squeeze(img)
    np.save(img_path, img)


def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float32) # np.float64
    img2 = img2.astype(np.float32) # np.float64
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    # return 20 * math.log10(255.0 / math.sqrt(mse))
    return 20 * math.log10(1.0 / math.sqrt(mse))



### 小波变换 ----------------------------------------------------------------
class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False  # 信号处理，非卷积运算，不需要进行梯度求导
    def forward(self, x):
        return dwt_init(x)

def dwt_init(x):
    x01 = x[:, :, :, 0::2, :] / 4
    x02 = x[:, :, :, 1::2, :] / 4
    x1 = x01[:, :, :, :, 0::2]
    x2 = x02[:, :, :, :, 0::2]
    x3 = x01[:, :, :, :, 1::2]
    x4 = x02[:, :, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    return torch.cat((x_LL, x_HL, x_LH, x_HH), dim=2)  # 沿channel维度拼接


### 逆小波变换 ----------------------------------------------------------------
class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return iwt_init(x)

def iwt_init(x):
    r = 2
    B, N, C, H, W = x.size()
    out_channel = C // (r**2)
    out_height = r * H
    out_width = r * W

    x1 = x[:, :, 0:out_channel, :, :]  # LL 子带
    x2 = x[:, :, out_channel:2*out_channel, :, :]  # HL 子带
    x3 = x[:, :, 2*out_channel:3*out_channel, :, :]  # LH 子带
    x4 = x[:, :, 3*out_channel:4*out_channel, :, :]  # HH 子带

    h = torch.zeros([B, N, out_channel, out_height, out_width]).float().to(x.device)
    h[:, :, :, 0::r, 0::r] = x1 - x2 - x3 + x4
    h[:, :, :, 1::r, 0::r] = x1 - x2 + x3 - x4
    h[:, :, :, 0::r, 1::r] = x1 + x2 - x3 - x4
    h[:, :, :, 1::r, 1::r] = x1 + x2 + x3 + x4
    return h

