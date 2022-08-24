import os
import numpy as np
import cv2
import glob
import random
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader

import segmentation_models_pytorch as smp

from dataset import IDCardSegmentDataset
from augmentation import get_augmentation, get_preprocessing
from utils import get_class_infos, count_parameters, load_model


def main(opt):
    model = smp.FPN(encoder_name=opt.encoder,
                    encoder_weights=opt.encoder_weights,
                    classes=len(get_class_infos(opt.train_dir, opt.ignores_list)),
                    activation='sigmoid',)
    
    # if opt.pretrained is not None:
    #     model = load_model(opt.pretrained)

    print(f"Using FPN model with {opt.encoder} backbone, pretrained from {opt.encoder_weights}")
    print(f"-- Number of parameters: {count_parameters(model)}")

    preprocessing_fn = smp.encoders.get_preprocessing_fn(opt.encoder, opt.encoder_weights)

    train_dataset = IDCardSegmentDataset(opt.train_dir, ignores_list=opt.ignores_list,
                                        augmentation=get_augmentation(img_shape=(640, 640)),
                                        preprocessing=get_preprocessing(preprocessing_fn))
    valid_dataset = IDCardSegmentDataset(opt.valid_dir, ignores_list=opt.ignores_list,
                                        augmentation=get_augmentation(is_test=True, img_shape=(640, 640)),
                                        preprocessing=get_preprocessing(preprocessing_fn))

    train_loader = DataLoader(train_dataset, batch_size=opt.train_bs, shuffle=True, num_workers=12)
    valid_loader = DataLoader(valid_dataset, batch_size=opt.valid_bs, shuffle=False, num_workers=4)

    # model = load_model('/home/manhdq/ID_Card_Information_Extraction/line_detection/FPN/runs/info_segmentation.pth')

    loss = smp.utils.losses.DiceLoss()
    metrics = [smp.utils.metrics.IoU(threshold=0.5)]
    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=0.0001)
    ])

    # Create epoch runners
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=opt.device,
        verbose=True
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=opt.device,
        verbose=True
    )

    max_score = 0

    for i in range(0, opt.epochs):
        
        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)
        
        # do something (save model, change lr, etc.)
        if max_score < valid_logs['iou_score']:
            print(f"IoU score increase: {max_score} --> {valid_logs['iou_score']}")
            max_score = valid_logs['iou_score']
            torch.save(model, f'{opt.save_dir}/best_model.pth')
            print('Model saved!')
            
        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')
        elif i == 45:
            optimizer.param_groups[0]['lr'] = 1e-5 * 0.01
            print('Decrease decoder learning rate to 1e-5!')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a line detector')
    parser.add_argument('--train-dir', help='the dir to train data', required=True)
    parser.add_argument('--valid-dir', help='the dir to valid data', required=True)
    parser.add_argument('--save-dir', help='the dir to save progress', required=True)
    parser.add_argument('--encoder', default='resnet50', help='backbone for segmentation model')
    parser.add_argument('--encoder-weights', default='imagenet', help='pretrained weights')
    parser.add_argument('--pretrained', help='pretrained weight path for model')
    parser.add_argument('--train-bs', type=int, default=8, help='train batch size')
    parser.add_argument('--valid-bs', type=int,  default=4, help='validation batch size')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--no-gpu', action='store_true', help='disabled gpu')
    parser.add_argument('--epochs', type=int, default=40, help='num epochs')

    parser.add_argument('--ignores-list', nargs='+', default=['background'], help='ignored classes')

    opt = parser.parse_args()
    if not os.path.exists(opt.save_dir):
        os.makedirs(opt.save_dir, exist_ok=True)
    opt.device = 'cpu' if opt.no_gpu else 'cuda:1'

    print(opt)

    """ Seed and GPU setting """
    # print("Random Seed: ", opt.manualSeed)
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True
    cudnn.deterministic = True

    main(opt)