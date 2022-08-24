import torch.nn as nn


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
            cfg_ = cfg_ | default_args
            cfg_type = cfg_.pop('type')
            eval(cfg_type)(**cfg_)
        return nn.Sequential(*modules)
    else:
        cfg = cfg | default_args
        cfg_type = cfg.pop('type')
        return eval(cfg_type)(**cfg)


def build_backbone(cfg):
    """Build backbone."""
    return build(cfg)


def build_neck(cfg):
    """Build neck."""
    return build(cfg)


def build_roi_extractor(cfg):
    """Build roi extractor."""
    return build(cfg)


def build_shared_head(cfg):
    """Build shared head."""
    return build(cfg)


def build_head(cfg):
    """Build head."""
    return build(cfg)


def build_loss(cfg):
    """Build loss."""
    return build(cfg)


def build_detector(cfg, train_cfg=None, test_cfg=None):
    """Build detector."""
    return build(cfg, dict(train_cfg=train_cfg, test_cfg=test_cfg))