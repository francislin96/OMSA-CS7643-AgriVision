import numpy as np
import cv2
import torch
from src.datasets.fixmatch_datasets import AgDataset, get_datasets
from src.utils.transforms import strong_tfms, weak_tfms, null_tfms
from src.datasets.dataloaders import get_dataloaders

transform_dict = {
    'strong': strong_tfms,
    'weak': weak_tfms,
    'val': null_tfms,
    'test': null_tfms
}
ds_dict = get_datasets(
    train_l_dir='./data/images_2021/train', 
    train_u_dir='./data/images_2024/train', 
    val_dir='./data/images_2021/val', 
    test_dir='./data/images_2021/test', 
    transform_dict=transform_dict, 
    ssl=True
)

(train_l_loader, train_u_loader), val_loader, test_loader = get_dataloaders(
    train_l_ds=ds_dict['train']['labeled'],
    val_ds=ds_dict['val'],
    train_u_ds=ds_dict['train']['unlabeled']
)

# for batch in val_loader:
#     img, target = batch
#     print(img.shape)
    # print(len(batch))
    # print(len(batch[0]))
    # print(len(batch[0][0]))
    # print(batch[0][0].shape)

for batch in train_u_loader:
    # print(batch.shape)
    weak_img, strong_img = batch
    print(weak_img.shape)
    print(strong_img.shape)