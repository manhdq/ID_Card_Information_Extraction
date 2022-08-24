from .iou2d_calculator import BboxOverlaps2D


def build(cfg, default_args=None):
    cfg = cfg.copy()
    cfg_type = cfg.pop('type')
    return eval(cfg_type)(**cfg)


def build_iou_calculator(cfg):
    """Build backbone."""
    return build(cfg)