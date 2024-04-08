#!/usr/bin/env python
# coding: utf-8


import numpy as np
from PIL import Image
import os


test_folder_path='<folder path>/Agriculture-Vision-2021/2021/test/images/rgb/'
test_imgs_path = os.listdir(test_folder_path)
test_full_paths=[test_folder_path+img for img in test_imgs_path]
val_folder_path='<folder path>/Agriculture-Vision-2021/2021/val/images/rgb/'
val_imgs_path = os.listdir(val_folder_path)
val_full_paths=[val_folder_path+img for img in val_imgs_path]
train_folder_path='<folder path>/Agriculture-Vision-2021/2021/train/images/rgb/'
train_imgs_path = os.listdir(train_folder_path)
train_full_paths=[train_folder_path+img for img in train_imgs_path]

def RGB_mean_and_std (full_paths):
    
    #return the list of mean and std 
    #1.calculate the mean of each channels
    sum_r=0
    sum_g=0
    sum_b=0
    for i in range(len(full_paths)):
        img_path=full_paths[i]
        img=np.array(Image.open(img_path))
        img=np.transpose(img,(2,0,1))
        sum_r+=np.sum(img[0,:,:])
        sum_g+=np.sum(img[1,:,:])
        sum_b+=np.sum(img[2,:,:])
    mean_r=sum_r/len(full_paths)/(img.shape[1]*img.shape[2])
    mean_g=sum_g/len(full_paths)/(img.shape[1]*img.shape[2])
    mean_b=sum_b/len(full_paths)/(img.shape[1]*img.shape[2])
    
    #2.calculate the std of each channels
    std_r=0
    std_g=0
    std_b=0
    for i in range(len(full_paths)):
        img_path=full_paths[i]
        img=np.array(Image.open(img_path))
        img=np.transpose(img,(2,0,1))
        std_r += np.sum((img[0,:,:]-mean_r)**2)
        std_g += np.sum((img[1,:,:]-mean_g)**2)
        std_b += np.sum((img[2,:,:]-mean_b)**2)
    std_r=(std_r/len(full_paths)/(img.shape[1]*img.shape[2]))**0.5
    std_g=(std_g/len(full_paths)/(img.shape[1]*img.shape[2]))**0.5
    std_b=(std_b/len(full_paths)/(img.shape[1]*img.shape[2]))**0.5
    
    return [(mean_r,mean_g,mean_b),(std_r,std_g,std_b)]

