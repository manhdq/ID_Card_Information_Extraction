from .assign_result import AssignResult
from .atss_assigner import ATSSAssigner
from .base_assigner import BaseAssigner


def build(cfg, default_args=None):
    cfg = cfg.copy()
    cfg_type = cfg.pop('type')
    return eval(cfg_type)(**cfg)


def build_assigner(cfg):
    """Build backbone."""
    return build(cfg)