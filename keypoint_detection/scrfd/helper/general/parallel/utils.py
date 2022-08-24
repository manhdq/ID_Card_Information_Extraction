from torch import nn
from torch.nn.parallel import DataParallel, DistributedDataParallel


MODULE_WRAPPERS = [DataParallel, DistributedDataParallel]

def is_module_wrapper(module: nn.Module) -> bool:
    """Check if a module is a module wrapper.
    The following 3 modules in MMCV (and their subclasses) are regarded as
    module wrappers: DataParallel, DistributedDataParallel,
    MMDistributedDataParallel (the deprecated version). You may add you own
    module wrapper by registering it to mmcv.parallel.MODULE_WRAPPERS or
    its children registries.
    Args:
        module (nn.Module): The module to be checked.
    Returns:
        bool: True if the input module is a module wrapper.
    """

    def is_module_in_wrapper(module, module_wrapper):
        module_wrappers = tuple(module_wrapper)
        if isinstance(module, module_wrappers):
            return True
        return False

    return is_module_in_wrapper(module, MODULE_WRAPPERS)