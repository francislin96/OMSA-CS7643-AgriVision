#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from PIL import Image
import os


# In[ ]:


test_folder_path='<file path>/Agriculture-Vision-2021/2021/test/images/nir/'
test_imgs_path = os.listdir(test_folder_path)
test_full_paths=[test_folder_path+img for img in test_imgs_path]
val_folder_path='<file path>/Agriculture-Vision-2021/2021/val/images/nir/'
val_imgs_path = os.listdir(val_folder_path)
val_full_paths=[val_folder_path+img for img in val_imgs_path]
train_folder_path='<file path>/Agriculture-Vision-2021/2021/train/images/nir/'
train_imgs_path = os.listdir(train_folder_path)
train_full_paths=[train_folder_path+img for img in train_imgs_path]


def NIR_mean_and_std (full_paths):
    
    #return the list of mean and std 
    #1.calculate the mean 
    sum_nir=0
    for i in range(len(full_paths)):
        img_path=full_paths[i]
        img=np.array(Image.open(img_path))
        sum_nir+=np.sum(img)
    mean_nir=sum_nir/len(full_paths)/(img.shape[0]*img.shape[1])
 
    #2.calculate the std 
    std_nir=0
    for i in range(len(full_paths)):
        img_path=full_paths[i]
        img=np.array(Image.open(img_path))
        std_nir += np.sum((img-mean_nir)**2)
    std_nir=(std_nir/len(full_paths)/(img.shape[0]*img.shape[1]))**0.5
    
    return [mean_nir,std_nir]

