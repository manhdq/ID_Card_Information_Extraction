import argparse
import os
import sys
import pickle
import numpy as np
import datetime
import warnings
import xml.etree.ElementTree as ET

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
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--data-dir', help='data images direcory for xml infos generation')
    parser.add_argument('--raw-xml-file', help='path to raw xml file')
    parser.add_argument('--save-xml-file', help='path to xml file', required=True)
    parser.add_argument('--save-visual-results-dir', help='directory for visual results')
    parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
    parser.add_argument('--conf-thr', help='confidence threshold', type=float, default=0.6)
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1280, 960],
        help='input image size')
    args = parser.parse_args()

    base_save_xml_path = args.save_xml_file.rsplit(os.sep, 1)[0]
    os.makedirs(base_save_xml_path, exist_ok=True)
    
    if args.save_visual_results_dir is not None:
        os.makedirs(args.save_visual_results_dir, exist_ok=True)

    return args


def main():
    args = parse_args()
    assert args.save_xml_file.split('.')[-1] == 'xml'

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None

    if args.data_dir is not None:
        cfg.data.test.img_prefix = os.path.join(args.data_dir, 'images')
        cfg.data.test.ann_file = os.path.join(args.data_dir, 'annotations.txt')

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    pipelines = cfg.data.test.pipeline
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
    samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
    if samples_per_gpu > 1:
        # Replace 'ImageToTensor' to 'DefaultFormatBundle'
        cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    device = torch.device("cpu" if args.cpu else "cuda")

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    if 'CLASSES' in checkpoint['meta']:
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    model = model.to(device)

    preds = {}

    model.eval()
    dataset = data_loader.dataset
    for i, data in enumerate(dataset):
        #print(img.shape)
        # img = img[:,:,:args.shape[1],:args.shape[0]]
        # img = img.to(device)
        # print(data['img_metas'][0]._data['filename'])
        # img_path = data['img_metas'][0]._data['filename']
        file_path = data['img_metas'][0]._data['filename']
        image_name = file_path.rsplit(os.sep, 1)[-1]

        data['img_metas'] = [[data['img_metas'][0]._data]]
        data['img'][0] = data['img'][0].unsqueeze(0)

        if not args.cpu:
            data = scatter(data, [device])[0]
        # print(data['img'][0].shape)
        # exit(0)

        result = model(return_loss=False, rescale=True, inference_end=True, **data)[0]
        bboxes_result, kps_result = result
        if args.save_visual_results_dir:
            result_pyplot_bboxes_kps_save(model, file_path, bboxes_result, kps_result, score_thr=0.5,
                                        save_path=os.path.join(args.save_visual_results_dir, file_path.split(os.sep)[-1]))

        preds[image_name] = kps_result

        # with torch.no_grad():
        #     ta = datetime.datetime.now()
        #     result = model.feature_test(img)
        #     tb = datetime.datetime.now()
        #     print('cost:', (tb-ta).total_seconds())
    
    root = ET.parse(args.raw_xml_file)
    tree = root.getroot()
    for idx in range(len(tree)):
        if tree[idx].tag != 'image':
            continue
        
        # print(tree[idx])
        # print(ET.SubElement(tree, 'image').attrib)
        # exit(0)
        print(tree[idx].attrib['name'])
        image_name = tree[idx].attrib['name']

        width = tree[idx].attrib['width']
        height = tree[idx].attrib['height']

        kps_result = preds[image_name][0]
        kps_result = kps_result[kps_result[:, -1] > args.conf_thr]
        for kps in kps_result:
            kps = kps[:-1].reshape(-1, 2)
            assert kps.shape[0] == 4
            kps_str_list = []
            for kp in kps:
                kps_str_list.append(f"{kp[0]:.2f},{kp[1]:.2f}")
            kps_str = ";".join(kps_str_list)

            polygon = ET.SubElement(tree[idx], 'polygon')
            polygon.set('label', 'CCCD')
            polygon.set('occluded', '0')
            polygon.set('source', 'manual')
            polygon.set('points', kps_str)
            polygon.set('z_order', "0")

    with open(args.save_xml_file, 'wb') as f:
        root.write(f)



if __name__ == '__main__':
    main()