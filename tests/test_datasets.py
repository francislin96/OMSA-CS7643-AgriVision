import numpy as np
import cv2
import torch
from src.datasets.fixmatch_datasets import AgDataset, get_datasets
from src.utils.transforms import strong_tfms, weak_tfms, null_tfms

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

train_u_ds = ds_dict['train']['unlabeled']
cv2.namedWindow('test', cv2.WINDOW_NORMAL)
idx = np.random.choice(np.arange(0, 30000), size=20)
for i in idx:
    print(i)
    print(train_u_ds.nir_names[i] == train_u_ds.rgb_names[i])
    weak_img, strong_img = train_u_ds[i]
    cv2.imshow('test', weak_img)
    cv2.waitKey(0)
    cv2.imshow('test', strong_img)
    cv2.waitKey(0)

cv2.destroyAllWindows()

train_l_ds = ds_dict['train']['labeled']
idx = np.random.choice(np.arange(0, 30000), size=20)
for i in idx:
    print(i)
    print(train_l_ds.nir_names[i] == train_l_ds.rgb_names[i])
    img, mask = train_l_ds[i]
    cv2.imshow('test', img)
    cv2.waitKey(0)

cv2.destroyAllWindows()

val_ds = ds_dict['val']
idx = np.random.choice(np.arange(0, 15000), size=20)
for i in idx:
    print(i)
    print(val_ds.nir_names[i] == val_ds.rgb_names[i])
    # stacked, mask = train_ds[i]
    img, mask = val_ds[i]
    img = np.moveaxis(img.cpu().numpy(), source=0, destination=2)
    print(img.shape)

    cv2.imshow('test', img)
    cv2.waitKey(0)

cv2.destroyAllWindows()