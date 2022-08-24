import warnings

import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
from ...general.ops import RoIPool
from ...general.parallel import collate, scatter
from ...general.runner import load_checkpoint
from ...general.utils import Config, DictAction
from ...general.image import imread, bgr2rgb

from ..core import get_classes
from ..datasets.pipelines import Compose
# from mmdet.datasets.pipelines import Compose
from  ..models import build_detector
# from mmdet.models import build_detector


def init_detector(config, checkpoint=None, device='cuda:0', cfg_options=None):
    """Initialize a detector from config file.
    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        cfg_options (dict): Options to override some settings in the used
            config.
    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, str):
        config = Config.fromfile(config)
    elif not isinstance(config, Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    config.model.pretrained = None
    model = build_detector(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        map_loc = 'cpu' if device == 'cpu' else None
        checkpoint = load_checkpoint(model, checkpoint, map_location=map_loc)
        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


class LoadImage(object):
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.
        Args:
            results (dict): A result dict contains the file name
                of the image to be read.
        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = imread(results['img'])
        results['img'] = img
        results['img_fields'] = ['img']
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def inference_detector(model, img):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # prepare data
    if isinstance(img, np.ndarray):
        # directly add img
        data = dict(img=img)
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
    else:
        # add information into dict
        data = dict(img_info=dict(filename=img), img_prefix=None)
    # build the data pipeline
    test_pipeline = Compose(cfg.data.test.pipeline)

    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'
        # just get the actual data from DataContainer
        data['img_metas'] = data['img_metas'][0].data

    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, inference_end=True, **data)[0]
    return result


def inference_batch_detector(model, imgs):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        If imgs is a str, a generator will be returned, otherwise return the
        detection results directly.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # prepare data
    datas = []
    for img in imgs:
        if isinstance(img, np.ndarray) or isinstance(img[0], np.ndarray):
            # directly add img
            data = dict(img=img)
            cfg = cfg.copy()
            # set loading pipeline type
            cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)

        # build the data pipeline
        test_pipeline = Compose(cfg.data.test.pipeline)
        data = test_pipeline(data)
        datas.append(data)
    datas = collate(datas, samples_per_gpu=len(datas))
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        datas = scatter(datas, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'
        # just get the actual data from DataContainer
        datas['img_metas'] = datas['img_metas'][0].data

    # print(data)
    # exit(0)
    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, inference_end=True, **datas)
    return result


async def async_inference_detector(model, img):
    """Async inference image(s) with the detector.
    Args:
        model (nn.Module): The loaded detector.
        img (str | ndarray): Either image files or loaded images.
    Returns:
        Awaitable detection results.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # prepare data
    if isinstance(img, np.ndarray):
        # directly add img
        data = dict(img=img)
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
    else:
        # add information into dict
        data = dict(img_info=dict(filename=img), img_prefix=None)
    # build the data pipeline
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]

    # We don't restore `torch.is_grad_enabled()` value during concurrent
    # inference since execution can overlap
    torch.set_grad_enabled(False)
    result = await model.aforward_test(rescale=True, **data)
    return result


def show_result_pyplot(model,
                       img,
                       result,
                       score_thr=0.3,
                       fig_size=(15, 10),
                       title='result',
                       block=True):
    """Visualize the detection results on the image.
    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
        title (str): Title of the pyplot figure.
        block (bool): Whether to block GUI.
    """
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(img, result, score_thr=score_thr, show=False)
    plt.figure(figsize=fig_size)
    plt.imshow(bgr2rgb(img))
    plt.title(title)
    plt.tight_layout()
    plt.show(block=block)


def result_pyplot_save(model,
                    img,
                    result,
                    score_thr=0.3,
                    title='result',
                    block=True):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
        title (str): Title of the pyplot figure.
        block (bool): Whether to block GUI.
    """
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(img, result, score_thr=score_thr, show=False)
    cv2.imwrite('test.jpg', img)


def result_pyplot_bboxes_kps_save(model,
                    img,
                    bboxes_result,
                    kps_result,
                    score_thr=0.3,
                    title='result',
                    block=True):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
        title (str): Title of the pyplot figure.
        block (bool): Whether to block GUI.
    """
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(img, bboxes_result, score_thr=score_thr, show=False)
    if isinstance(bboxes_result, tuple):
        bboxes_result = bboxes_result[0]
    bboxes = np.vstack(bboxes_result)
    inds = np.where(bboxes[:, -1] > score_thr)[0]
    kps_result = kps_result[0][inds]
    kps_result = kps_result[:, :-1].reshape(-1, 4, 2)
    img = cv2.polylines(img, kps_result.astype(np.int32), True, (255, 0, 0,), 2)
    cv2.imwrite('test.jpg', img)


def result_pyplot_bboxes_kps_save(model,
                    img,
                    bboxes_result,
                    kps_result,
                    score_thr=0.3,
                    title='result',
                    block=True,
                    save_path='test.jpg'):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        fig_size (tuple): Figure size of the pyplot figure.
        title (str): Title of the pyplot figure.
        block (bool): Whether to block GUI.
    """
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(img, bboxes_result, score_thr=score_thr, show=False)
    if isinstance(bboxes_result, tuple):
        bboxes_result = bboxes_result[0]
    bboxes = np.vstack(bboxes_result)
    inds = np.where(bboxes[:, -1] > score_thr)[0]
    kps_result = kps_result[0][inds]
    kps_result = kps_result[:, :-1].reshape(-1, 4, 2)
    img = cv2.polylines(img, kps_result.astype(np.int32), True, (255, 0, 0,), 2)
    cv2.imwrite(save_path, img)