import torch.nn as nn

from .mobilenet import MobileNetV1


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


def build_backbone(cfg):
    """Build backbone."""
    return build(cfg)