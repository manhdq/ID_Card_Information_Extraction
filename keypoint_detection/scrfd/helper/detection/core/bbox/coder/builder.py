from .base_bbox_coder import BaseBBoxCoder
from .delta_xywh_bbox_coder import DeltaXYWHBBoxCoder


def build(cfg, default_args=None):
    cfg = cfg.copy()
    cfg_type = cfg.pop('type')
    return eval(cfg_type)(**cfg)


def build_bbox_coder(cfg):
    """Build backbone."""
    return build(cfg)