# Weak and strong augmentations for images

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


def base_train_tfms():
    transforms = A.Compose(
        [
            ## Francis fill in here
            ToTensorV2()
        ]
    )

    return transforms


def base_val_tfms():
    transforms = A.Compose(
        [
            ## Francis fill in here
            ToTensorV2()
        ]
    )

    return transforms


def weak_tfms():
    transforms = A.Compose([
        A.Affine(translate_percent=(1, 12.5), p=1.0),
        A.HorizontalFlip(p=0.5),
        ToTensorV2(p=1.0)
    ])

    return transforms


def strong_tfms():
    transforms = A.Compose([
        A.RandomBrightnessContrast(),
        A.ColorJitter(),
        A.Posterize(),
        A.SafeRotate(),
        A.Sharpen((0.05, 0.95)),
        A.Affine(translate_percent=12.5, shear=(-30, 30)),
        A.Solarize(),
        A.GaussianBlur()
    ])

    return transforms