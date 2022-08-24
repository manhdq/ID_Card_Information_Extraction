from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....general.utils import TORCH_VERSION, digit_version


ACTIVATION_LAYERS = {
    'ReLU': nn.ReLU,
    'LeakyReLU': nn.LeakyReLU,
    'PReLU': nn.PReLU,
    'RReLU': nn.RReLU,
    'ReLU6': nn.ReLU6,
    'ELU': nn.ELU,
    'Sigmoid': nn.Sigmoid,
    'Tanh': nn.Tanh
}

class Clamp(nn.Module):
    """Clamp activation layer.
    This activation function is to clamp the feature map value within
    :math:`[min, max]`. More details can be found in ``torch.clamp()``.
    Args:
        min (Number | optional): Lower-bound of the range to be clamped to.
            Default to -1.
        max (Number | optional): Upper-bound of the range to be clamped to.
            Default to 1.
    """

    def __init__(self, min: float = -1., max: float = 1.):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x) -> torch.Tensor:
        """Forward function.
        Args:
            x (torch.Tensor): The input tensor.
        Returns:
            torch.Tensor: Clamped tensor.
        """
        return torch.clamp(x, min=self.min, max=self.max)


def build_activation_layer(cfg: Dict) -> nn.Module:
    """Build activation layer.
    Args:
        cfg (dict): The activation layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an activation layer.
    Returns:
        nn.Module: Created activation layer.
    """
    cfg = cfg.copy()
    cfg_type = cfg.pop('type')
    return ACTIVATION_LAYERS.get(cfg_type)(**cfg)