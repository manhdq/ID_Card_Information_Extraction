import os
import cv2
import numpy as np

import albumentations as A

import segmentation_models_pytorch as smp


def get_augmentation(is_test=False, img_shape=(500, 300)):
    if not is_test:  # training
        transform = [
            A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=0.1, shift_limit=0.1, p=1, border_mode=0),
            A.PadIfNeeded(min_height=img_shape[1], min_width=img_shape[0], always_apply=True, border_mode=0),
            # A.IAAAdditiveGaussianNoise(p=0.5),
            # A.IAAPerspective(p=0.1),

            A.OneOf(
                [
                    A.CLAHE(p=1),
                    A.RandomBrightness(p=1),
                    A.RandomGamma(p=1),
                ],
                p=0.5
            ),

            A.OneOf(
                [
                    # A.IAASharpen(p=1),
                    A.Blur(blur_limit=3, p=1),
                    A.MotionBlur(blur_limit=3, p=1),
                ],
                p=0.5,
            ),

            A.OneOf(
                [
                    A.RandomContrast(p=1),
                ],
                p=0.5
            ),
        ]
    else:
        transform = [
            A.PadIfNeeded(min_height=img_shape[1], min_width=img_shape[0], always_apply=True, border_mode=0),
        ]
    
    return A.Compose(transform)


def to_tensor(x, **kwargs):
    if len(x.shape) == 3:
        return x.transpose(2, 0, 1).astype('float32')

    return x.astype('float32')

def get_preprocessing_fn(encoder, encoder_weights_name):
    ENCODER = encoder
    ENCODER_WEIGHTS = encoder_weights_name
    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

    return preprocessing_fn

def get_preprocessing(preprocessing_fn):
    _transform = [
        A.Lambda(image=preprocessing_fn),
        A.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return A.Compose(_transform)