import inspect
from typing import List

import torch


def register_torch_optimizers() -> List:
    torch_optimizers = {}
    for module_name in dir(torch.optim):
        if module_name.startswith('__'):
            continue
        _optim = getattr(torch.optim, module_name)
        if inspect.isclass(_optim) and issubclass(_optim,
                                                  torch.optim.Optimizer):
            torch_optimizers[module_name] = _optim
    return torch_optimizers


TORCH_OPTIMIZERS = register_torch_optimizers()