# Weak and strong augmentations for images

import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def train_tfms():
    transforms = A.Compose(
        [
            A.RandomBrightnessContrast(),
            A.Posterize(),
            A.SafeRotate(),
            A.Sharpen((0.05, 0.95)),
            A.Affine(translate_percent=(-.125, .125), shear=(-15, 15)),
            A.Solarize(),
            A.GaussianBlur(),
            ToTensorV2(p=1.0)
        ], additional_targets={'mask': 'image'}
    )

    return transforms


def weak_tfms():
    transforms = A.Compose([
        A.Affine(translate_percent=(-.125, .125), p=1.0),
        A.HorizontalFlip(p=0.5),
        ToTensorV2(p=1.0)
    ])

    return transforms


def strong_tfms():
    transforms = A.Compose([
        A.RandomBrightnessContrast(),
        A.Posterize(),
        A.SafeRotate(),
        A.Sharpen((0.05, 0.95)),
        A.Affine(translate_percent=(-.125, .125), shear=(-15, 15)),
        A.Solarize(),
        A.GaussianBlur(),
        ToTensorV2(p=1.0)
    ])

    return transforms

def null_tfms():
    transforms = A.Compose([
        ToTensorV2(p=1.0)
    ], additional_targets={'mask': 'image'})

    return transforms

# def test_collate_fn(batch):
#     """
#     Handles batches of varying sizes for the Dataloader
#     Parameters:
#         The batch of images passed from a Dataset in a Dataloader
#     Returns:
#         A zipped tuple of a given batch
#     """
#     return tuple(zip(*batch))

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