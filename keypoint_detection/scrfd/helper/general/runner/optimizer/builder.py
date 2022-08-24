import copy
import inspect
from typing import Dict, List

import torch

from .default_constructor import DefaultOptimizerConstructor


def build_optimizer_constructor(cfg: Dict):
    cfg = cfg.copy()
    cfg_type = cfg.pop('type')
    return eval(cfg_type)(**cfg)


def build_optimizer(model, cfg: Dict):
    optimizer_cfg = copy.deepcopy(cfg)
    constructor_type = optimizer_cfg.pop('constructor',
                                         'DefaultOptimizerConstructor')
    paramwise_cfg = optimizer_cfg.pop('paramwise_cfg', None)
    optim_constructor = build_optimizer_constructor(
        dict(
            type=constructor_type,
            optimizer_cfg=optimizer_cfg,
            paramwise_cfg=paramwise_cfg))
    optimizer = optim_constructor(model)
    return optimizer