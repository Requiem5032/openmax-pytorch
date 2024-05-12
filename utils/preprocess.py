import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

from albumentations import (
    ShiftScaleRotate, RandomRotate90,
    Transpose, ShiftScaleRotate,
    Flip, Compose,
    Normalize, Resize,
    DualTransform
)


class Presize(DualTransform):
    """Randomly zoom into a part of the input.
    Args:
        zoom_limit (int): the maximum zoom that can be applied
        p (float): probability of applying the transform. Default: 1.
    Targets:
        image, mask, bboxes, keypoints
    Image types:
        uint8, float32
    """

    def __init__(self, zoom_limit, always_apply=True, p=1):
        super(Presize, self).__init__(always_apply, p)
        assert zoom_limit >= 1, 'Zoom limit should be greater than 1'
        self.zoom_limit = zoom_limit

    def get_params(self):
        new_size = np.random.uniform(1./self.zoom_limit, 1.)
        top = np.random.uniform(0, 1 - new_size)
        left = np.random.uniform(0, 1 - new_size)
        return {"new_size": new_size, "top": top, "left": left}

    def apply(self, img, new_size, top, left, **params):
        new_size_px = np.round(img.shape[0]*new_size).astype('int')
        top_px = np.round(img.shape[0]*top).astype('int')
        left_px = np.round(img.shape[1]*left).astype('int')
        return img[top_px:top_px+new_size_px, left_px:left_px+new_size_px]

    def apply_to_bbox(self, bbox, **params):
        # Bounding box coordinates are scale invariant
        raise

    def apply_to_keypoint(self, keypoint, **params):
        raise

    def get_transform_init_args_names(self):
        return ("zoom_limit")


def alb_transform_train(imsize=256, p=0.2):
    albumentations_transform = Compose([
        Presize(zoom_limit=2.),
        Resize(imsize, imsize),
        RandomRotate90(),
        Flip(),
        Transpose(),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1,
                         rotate_limit=45, p=0.5),
        Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ], p=1)
    return albumentations_transform


def alb_transform_test(imsize=256, p=1):
    albumentations_transform = Compose([
        Resize(imsize, imsize),
        Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ], p=1)
    return albumentations_transform
