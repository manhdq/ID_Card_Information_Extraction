import argparse
import os
import sys
import pickle
import numpy as np
import datetime
import warnings

import torch

FILE = os.path.dirname(__file__)
sys.path.append(os.path.join(FILE, '..'))
from helper.general.utils import Config, DictAction
from helper.general.cnn import fuse_conv_bn
from helper.general.parallel import MMDataParallel, MMDistributedDataParallel, scatter
from helper.general.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from helper.detection.apis import multi_gpu_test, single_gpu_test, inference_detector, result_pyplot_bboxes_kps_save
from helper.detection.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from helper.detection.models import build_detector
from helper.detection.core.evaluation import wider_evaluation, wider_kpts_evaluation, get_widerface_gts
# from helper.detection.apis import inference_detector, init_detector, show_result_pyplot, \
#                     result_pyplot_save, result_pyplot_bboxes_kps_save
#from torch.utils import mkldnn as mkldnn_utils


def parse_args():
    # Usage: python tools/benchmark.py configs/scrfd/scrfd_cccd_500m_bnkps_dummy.py \
    #                        work_dirs/scrfd_cccd_500m_bnkps_dummy/best3.pth
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1280, 960],
        help='input image size')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()


    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None

    # in case the test dataset is concatenated
    if isinstance(cfg.data.infer, dict):
        cfg.data.infer.test_mode = True
    elif isinstance(cfg.data.infer, list):
        for ds_cfg in cfg.data.infer:
            ds_cfg.test_mode = True

    pipelines = cfg.data.infer.pipeline
    for pipeline in pipelines:
        if pipeline.type=='MultiScaleFlipAug':
            #pipeline.img_scale = (640, 640)
            pipeline.img_scale = None
            pipeline.scale_factor = 1.0
            transforms = pipeline.transforms
            for transform in transforms:
                if transform.type=='Pad':
                    #transform.size = pipeline.img_scale
                    transform.size = None
                    transform.size_divisor = 1
    #print(cfg.data.test.pipeline)
    distributed = False

    # build the dataloader
    samples_per_gpu = cfg.data.infer.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.infer.pipeline = replace_ImageToTensor(cfg.data.infer.pipeline)
    dataset = build_dataset(cfg.data.infer)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    device = torch.device("cpu" if args.cpu else "cuda:1")

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.val_cfg)
    fp16_cfg = cfg.get('fp16', None)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    model = model.to(device)

    preds = {}
    gts = {}

    model.eval()
    dataset = data_loader.dataset
    for i, data in enumerate(dataset):
        #print(img.shape)
        # img = img[:,:,:args.shape[1],:args.shape[0]]
        # img = img.to(device)
        # print(data['img_metas'][0]._data['filename'])
        # img_path = data['img_metas'][0]._data['filename']
        filename = data['img_metas'][0]._data['filename']
        data['img_metas'] = [[data['img_metas'][0]._data]]
        data['img'][0] = data['img'][0].unsqueeze(0)
        gt_bboxes = data.pop('gt_bboxes')
        gt_labels = data.pop('gt_labels')
        gt_bboxes_ignore = data.pop('gt_bboxes_ignore')
        gt_keypointss = data.pop('gt_keypointss')

        if not args.cpu:
            data = scatter(data, [device])[0]
        # print(data['img'][0].shape)
        # exit(0)

        result = model(return_loss=False, rescale=True, inference_end=True, **data)[0]
        bboxes_result, kps_result = result
        result_pyplot_bboxes_kps_save(model, filename, bboxes_result, kps_result, score_thr=0.5,
                                    save_path=os.path.join('demo_cccd', filename.split(os.sep)[-1]))

        preds[filename] = kps_result
        gts[filename] = gt_keypointss

        # with torch.no_grad():
        #     ta = datetime.datetime.now()
        #     result = model.feature_test(img)
        #     tb = datetime.datetime.now()
        #     print('cost:', (tb-ta).total_seconds())
    for iou_th in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        ap, kious = wider_kpts_evaluation(preds, gts, iou_thresh=iou_th)
            


if __name__ == '__main__':
    main()