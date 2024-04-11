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
# ds_dict = get_datasets(
#     train_l_dir='./data/images_2021/train', 
#     train_u_dir='./data/images_2024/train', 
#     val_dir='./data/images_2021/val', 
#     test_dir='./data/images_2021/test', 
#     transform_dict=transform_dict, 
#     ssl=True
# )

ds_dict = get_datasets(
    train_l_dir='./data/dev_data/labeled/train', 
    train_u_dir='./data/dev_data/unlabeled/train', 
    val_dir='./data/dev_data/labeled/val', 
    test_dir='./data/dev_data/labeled/test', 
    transform_dict=transform_dict, 
    ssl=True
)
print(ds_dict)

train_l_ds = ds_dict['train']['labeled']
train_u_ds = ds_dict['train']['unlabeled']
val_ds = ds_dict['val']
test_ds = ds_dict['test']

print(len(train_l_ds))
print(len(train_u_ds))
print(len(val_ds))
# print(len(test_ds))

(train_l_loader, train_u_loader), val_loader, test_loader = get_dataloaders(
    train_l_ds=ds_dict['train']['labeled'],
    val_ds=ds_dict['val'],
    train_u_ds=ds_dict['train']['unlabeled'],
    batch_size=2
)

print(len(train_l_loader))
print(len(train_u_loader))

for batch in train_u_loader:
    # print(batch.shape)
    weak_img, strong_img = batch
    print(weak_img.shape)
    print(strong_img.shape)

for batch in train_l_loader:
    img, target = batch
    print(img.shape)
    print(target.shape)


# for batch_idx, batch in enumerate(
#         zip(train_l_loader, train_u_loader)
#     ):
#     print(batch_idx)
#     l_batch, u_batch = batch
#     l_img, l_mask = l_batch
#     print(l_img.shape)
#     print(l_mask.shape)

#     u_weak, u_strong = u_batch
#     print(u_weak.shape)
#     print(u_strong.shape)