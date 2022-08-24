import cv2
import os
import numpy as np

import torch

def letterbox(im, new_shape=(300, 500), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def get_class_infos(dir, ignores_list):
    if not dir.split('.')[-1] == 'txt':
        dir = os.path.join(dir, 'labelmap.txt')
        
    colors_dict = {}
    with open(os.path.join(dir), 'r') as fr:
        lines = fr.readlines()
        for line in lines[1:]:
            feat_infos = line.split(':')
            feat_name = feat_infos[0]
            if feat_name in ignores_list:
                continue
            feat_color = feat_infos[1].split(',')
            feat_color = [int(ch_color) for ch_color in feat_color]
            assert len(feat_color) == 3
            colors_dict[feat_name] = feat_color

    return colors_dict

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_model(path, map_location=None):
    return torch.load(path, map_location=map_location)

def get_min_max_x_y(coordinates):
    min_y, min_x, max_y, max_x = min(coordinates[:, 0]), min(coordinates[:, 1]), max(coordinates[:, 0]), \
                                 max(coordinates[:, 1])

    return min_x, min_y, max_x, max_y