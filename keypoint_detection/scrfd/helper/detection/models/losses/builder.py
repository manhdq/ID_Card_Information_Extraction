import torch.nn as nn

from .gfocal_loss import DistributionFocalLoss, QualityFocalLoss
from .iou_loss import (BoundedIoULoss, CIoULoss, DIoULoss, GIoULoss, IoULoss,
                       bounded_iou_loss, iou_loss)
from .smooth_l1_loss import SmoothL1Loss, L1Loss, l1_loss, smooth_l1_loss
from .utils import reduce_loss, weight_reduce_loss, weighted_loss


def build(cfg, default_args=None):
    """Build a module.
    Args:
        cfg (dict, list[dict]): The config of modules, is is either a dict
            or a list of configs.
        registry (:obj:`Registry`): A registry the module belongs to.
        default_args (dict, optional): Default arguments to build the module.
            Defaults to None.
    Returns:
        nn.Module: A built nn module.
    """
    if isinstance(cfg, list):
        modules = []
        for cfg_ in cfg:
            cfg_ = cfg_.copy()
            cfg_type = cfg_.pop('type')
            eval(cfg_type)(**cfg_)
        return nn.Sequential(*modules)
    else:
        cfg = cfg.copy()
        cfg_type = cfg.pop('type')
        return eval(cfg_type)(**cfg)


def build_loss(cfg):
    """Build neck."""
    return build(cfg)