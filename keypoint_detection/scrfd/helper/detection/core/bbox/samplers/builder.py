from .base_sampler import BaseSampler
from .sampling_result import SamplingResult
from .pseudo_sampler import PseudoSampler


def build(cfg, **kwargs):
    cfg = cfg.copy()
    cfg = cfg | kwargs
    cfg_type = cfg.pop('type')
    return eval(cfg_type)(**cfg)


def build_sampler(cfg, **kwargs):
    """Build backbone."""
    return build(cfg, **kwargs)