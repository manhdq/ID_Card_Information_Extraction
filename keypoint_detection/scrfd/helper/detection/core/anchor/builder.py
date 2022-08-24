from .point_generator import PointGenerator
from .anchor_generator import AnchorGenerator


def build(cfg, default_args=None):
    cfg = cfg.copy()
    cfg_type = cfg.pop('type')
    return eval(cfg_type)(**cfg)


def build_anchor_generator(cfg):
    """Build backbone."""
    return build(cfg)