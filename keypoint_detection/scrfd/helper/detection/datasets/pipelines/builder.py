# from .auto_augment import (AutoAugment, BrightnessTransform, ColorTransform,
#                            ContrastTransform, EqualizeTransform, Rotate, Shear,
#                            Translate)
from .formating import (Collect, DefaultFormatBundle, ImageToTensor,
                        ToDataContainer, ToTensor, Transpose, to_tensor)
from .instaboost import InstaBoost
from .loading import (LoadAnnotations, LoadImageFromFile, LoadImageFromWebcam,
                      LoadMultiChannelImageFromFiles, LoadProposals)
from .transforms import (Albu, CutOut, Expand, MinIoURandomCrop, Normalize,
                         Pad, PhotoMetricDistortion, RandomCenterCropPad,
                         RandomSquareCrop,
                         RandomCrop, RandomFlip, Resize, SegRescale)
# from .test_time_aug import MultiScaleFlipAug


def build(cfg, default_args=None):
    cfg = cfg.copy()
    cfg_type = cfg.pop('type')
    return eval(cfg_type)(**cfg)


def build_pipeline(cfg):
    """Build backbone."""
    return build(cfg)