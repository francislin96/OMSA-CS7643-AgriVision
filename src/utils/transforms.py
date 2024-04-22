# Weak and strong augmentations for images

import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def train_tfms(norm: dict=None) -> A.Compose:
    """Form a albumentations Compose function to transform an image and a mask for labeled train images.

    Args:
        norm (dict, optional): A dictionary with keys 'means' and 'std' and values are 4-tuples with dataset means and std. Defaults to None.

    Returns:
        A.Compose: An albumentations Compose function
    """

    if norm:
        means = norm['means']
        std = norm['std']
    else:
        means = (0, 0, 0, 0)
        std = (1, 1, 1, 1)

    transforms = A.Compose([
            A.Normalize(means, std),
            # A.ChannelShuffle(p=.2),
            # A.GaussNoise(),
            # A.RandomBrightnessContrast(p=.2),
            # A.SafeRotate(),
            # A.Sharpen((0.05, 0.95)),
            A.Affine(translate_percent=(-.05, .05), shear=(-5, 5)),
            # A.Solarize(),
            # A.GaussianBlur(),
            ToTensorV2(p=1.0)
    ], additional_targets={'target': 'mask', 'mask': 'mask'})

    return transforms


def weak_tfms(norm: dict=None) -> A.Compose:
    """Form a albumentations Compose function to weakly augment an image for the unlabeled set.

    Args:
        norm (dict, optional): A dictionary with keys 'means' and 'std' and values are 4-tuples with dataset means and std. Defaults to None.

    Returns:
        A.Compose: An albumentations Compose function
    """

    if norm:
        means = norm['means']
        std = norm['std']
    else:
        means = (0, 0, 0, 0)
        std = (1, 1, 1, 1)

    transforms = A.Compose([
        A.Normalize(means, std),
        A.Affine(translate_percent=(-.125, .125), p=1.0),
        A.HorizontalFlip(p=0.5),
        ToTensorV2(p=1.0)
    ])

    return transforms


def strong_tfms(norm: dict=None) -> A.Compose:
    """Form a albumentations Compose function to strongly augment an image for the unlabeled set.

    Args:
        norm (dict, optional): A dictionary with keys 'means' and 'std' and values are 4-tuples with dataset means and std. Defaults to None.

    Returns:
        A.Compose: An albumentations Compose function
    """

    if norm:
        means = norm['means']
        std = norm['std']
    else:
        means = (0, 0, 0, 0)
        std = (1, 1, 1, 1)

    transforms = A.Compose([
        A.Normalize(means, std),
        A.ChannelShuffle(),
        # A.GaussNoise(),
        A.RandomBrightnessContrast(),
        A.SafeRotate(),
        A.Sharpen((0.05, 0.95)),
        A.Affine(translate_percent=(-.125, .125), shear=(-15, 15)),
        # A.Solarize(),
        A.GaussianBlur(),
        ToTensorV2(p=1.0)
    ])

    return transforms

def null_tfms(norm: dict=None) -> A.Compose:
    """Form a albumentations Compose function to transform an image for the validation set.

    Args:
        norm (dict, optional): A dictionary with keys 'means' and 'std' and values are 4-tuples with dataset means and std. Defaults to None.

    Returns:
        A.Compose: An albumentations Compose function
    """

    if norm:
        means = norm['means']
        std = norm['std']
    else:
        means = (0, 0, 0, 0)
        std = (1, 1, 1, 1)

    transforms = A.Compose([
        A.Normalize(means, std, ),
        ToTensorV2(p=1.0)
    ], additional_targets={'target': 'mask', 'mask': 'mask'})

    return transforms


def collate_fn(batch):
    images, masks = zip(*batch)
    images = torch.stack(images, dim=0)  # Stacks into (N, C, H, W)
    masks = torch.stack(masks, dim=0)    # Stacks into (N, H, W)
    return images, masks

def unlab_collate_fn(batch):
    weak_img, strong_img = zip(*batch)
    weak_img = torch.stack(weak_img, dim=0)  # Stacks into (N, C, H, W)
    strong_img = torch.stack(strong_img, dim=0)    # Stacks into (N, C, H, W)
    return weak_img, strong_img