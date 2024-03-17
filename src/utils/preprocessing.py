# Image pre-processing functions

import numpy as np
import cv2
from glob import glob
from skimage.io import imread
import os
import matplotlib.pyplot as plt

def normalize_img(channel_list: list[np.ndarray]) -> tuple[np.ndarray]:
    """
    This function takes a list of numpy arrays and returns a color image and an NIR image
    Each array in the channel list should have the same dimensions.
    The channels need to be ordered as [NIR, RED, GREEN, BLUE].
    Preprocessing removes any bad pixel values, and clips to the 1st and 99th percentile of values to remove noise.
    The pixels are then normalized channel wise to the interval [0, 255].
    The color image is returned with shape (H, W, C) with channel in BGR format per OpenCV formatting.
    The NIR image is returned with shape (H, W, C) where C=1. 

    Args:
        channel_list (list[np.ndarray]): a list of 4 numpy arrays. Each array should have the same dimensions.

    Returns:
        tuple[np.ndarray]: A tuple of two numpy arrays, norm_bgr (H, W, C) and norm_nir (H, W, 1)
    """
    try:
        channel_stack = np.stack(channel_list, -1)
    except ValueError as e:
        print(e)
        return None, None
    for i in range(4):
        channel = channel_stack[:, :, i]

        # Replace special pixel values with channel minimum. Clip and normalize
        channel[np.where(channel == -10000)] = channel[np.where(channel != -10000)].min()
        qL, qU = np.percentile(channel, (1, 99))
        channel = np.clip(channel, qL, qU)
        channel_stack[:, :, i] = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F).astype(np.uint8)

    norm_bgr = cv2.cvtColor(channel_stack[:, :, 1:], cv2.COLOR_RGB2BGR)
    norm_nir = cv2.normalize(channel_stack[:, :, 0], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F).astype(np.uint8)

    return norm_bgr, norm_nir
