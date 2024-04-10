# Image pre-processing functions

import numpy as np
import cv2
import os
import json
import logging
from pathlib import Path

from typing import Union, Tuple

def stack_tifs(path: str) -> np.ndarray:
    """_summary_
    Reads in the NIR, R, G, B separate channels and stacks them as shape (H, W, C)
    Args:
        path (str): 

    Returns:
        np.ndarray: A 4 channel image in NIR-RGB format
    """

    logger = logging.getLogger('debug_logger')
    channels = ['nir.tif', 'red.tif', 'green.tif', 'blue.tif']
    stack_list = []
    for c in channels:
        try:
            channel = cv2.imread(os.path.join(path, c), -1)
            stack_list.append(channel)
        except cv2.error as e:
            logger.error(f"FILE: {path} - CHANNEL: {c}" + e)
            return None
    try:
        stacked = np.stack(stack_list, -1)
    except ValueError as e:
        print(e)
        return None

    return stacked

def normalize_img(img_stack: np.ndarray, percentiles: tuple=(1, 99)) -> Tuple[np.ndarray, np.ndarray]:
    """_summary_
    This function takes a 4 channel numpy array and returns a color image and an NIR image
    The channels need to be ordered as [NIR, RED, GREEN, BLUE].
    Preprocessing removes any bad pixel values, and clips to the values in percentiles to remove noise.
    The pixels are then normalized channel wise to the interval [0, 255].
    The color image is returned with shape (H, W, C) with channel in BGR format per OpenCV formatting.
    The NIR image is returned with shape (H, W, C) where C=1. 

    Args:
        img_stack (np.ndarray): a list of 4 numpy arrays. Each array should have the same dimensions.
        percentiles (tuple, optional): Min and Max percentiles to clip values to. Defaults to (1, 99).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple of two numpy arrays, norm_bgr (H, W, C) and norm_nir (H, W, 1)
    """

    for i in range(4):
        channel = img_stack[:, :, i]

        # Replace special pixel values with channel minimum. Clip and normalize
        channel[np.where(channel == -10000)] = channel[np.where(channel != -10000)].min()
        qL, qU = np.percentile(channel, (1, 99))
        channel = np.clip(channel, qL, qU)
        img_stack[:, :, i] = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F).astype(np.uint8)

    norm_bgr = cv2.cvtColor(img_stack[:, :, 1:], cv2.COLOR_RGB2BGR).astype(np.uint8)
    norm_nir = cv2.normalize(img_stack[:, :, 0], None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F).astype(np.uint8)

    return norm_bgr, norm_nir

def convert_boundary_to_mask(boundary_path: str, img: np.ndarray, img_name: str) -> np.ndarray:
    """Convert a set of polygon points to a mask of an image.

    Args:
        boundary_path (str): Path to boundary.json, contains Polygon coordinates. 
        img (np.ndarray): Image to be masked

    Returns:
        np.ndarray: A mask with values of 0 (exclude) or 255 (include)
    """
    logger = logging.getLogger('debug_logger')
    # helper function to check if points are in the right format, some are not
    def type_masker(mask: np.ndarray, boundary_path: str=boundary_path) -> bool:
        with open(boundary_path) as f:
            bound_dict = json.load(f)
        type = bound_dict['type']
        features = bound_dict['coordinates']
        if type=='Polygon':
            try:
                points = np.array(features, dtype=np.int32)
                mask = cv2.fillPoly(mask, points, 255)
            except ValueError as e:
                logger.debug(f'Boundary file for {img_name} is in the wrong format. Please correct it')
                print(e)
                mask = None
                # for f in features:
                #     points = np.array([f], dtype=np.int32)
                #     mask = cv2.fillPoly(mask, points, 255)
        elif type=='MultiPolygon':
            for f in features:
                try:
                    points = np.array(f, dtype=np.int32)
                    mask = cv2.fillPoly(mask, points, 255)
                except ValueError as e:
                    logger.debug(f'Boundary file for {img_name} is in the wrong format. Please correct it')
                    print(e)
                    mask = None
        
        return mask

    # Check for real path
    if not os.path.exists(boundary_path):
        raise(ValueError(f"The boundary path {boundary_path} does not exist."))
    
    # Initialize zero mask
    mask = np.zeros(img.shape[0:2], dtype=np.uint8)
    mask = type_masker(boundary_path=boundary_path, mask=mask)

    return mask

def apply_boundary_to_img(boundary_path: str, img: np.ndarray, img_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """Applies a boundary mask from boundary_path to an image.

    Args:
        boundary_path (str): Path to boundary.json, contains Polygon coordinates. 
        img (np.ndarray): Image to be masked

    Returns:
        Tuple[np.ndarray, np.ndarray]: Returns a tuple of the binary mask and the masked 3 channel or 1 channel image
    """
    
    # convert boundary to mask
    mask = convert_boundary_to_mask(boundary_path, img, img_name)
    if mask is None:
        return mask, img
    
    # apply mask to image
    masked_img = cv2.bitwise_and(img, img, mask=mask)

    return mask, masked_img

def crop_to_cnt(mask: np.ndarray, img: np.ndarray):
    """Crops an image to the smallest bounding box around the masked contours.

    Args:
        mask (np.ndarray): The single channel image mask
        img (np.ndarray): A three channel image of shape (H, W, C)

    Returns:
        _type_: Returns img cropped to [ymin:ymax, xmin:xmax, :]
    """
    cnt, _ = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    cnt = np.concatenate(cnt)
    x, y, w, h = cv2.boundingRect(cnt)

    cropped = img[y:y+h, x:x+w]

    return cropped

def split_img(input: Union[str, np.ndarray], crop_dim: tuple[int, int]=(512, 512)) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Takes input as an image path or a Numpy array, calculates how many images of size crop_dim can fit inside of input,
        removes a margin from the edges, and breaks image into patches of size crop_dim

    Args:
        input (Union[str, np.ndarray]): _description_
        crop_dim (tuple[int, int], optional): _description_. Defaults to (512, 512).

    Returns:
        Tuple[np.ndarray, Tuple[int, int]]: Returns a numpy array of non-overlapping image patches of shape (num_grid_h, num_grid_w, crop_dim[0], crop_dim[1], C)
            and a tuple of ints of the height margin and width margin removed from the raw image
    """

    # Validate input
    if type(input)==str:
        if not os.path.exists(input):
            raise(ValueError(f"Input {input} to cv2.imread does not exist. Please check path integrity."))
        
    elif type(input)==np.ndarray:
        if len(input.shape)==2:
            input = np.expand_dims(input,2)
        elif len(input.shape)==0 or len(input.shape)>3:
            raise(ValueError("Input Numpy array is the wrong shape. Input should be either a 2D or 3D array."))
    
    if not np.all([a >= b for a, b in zip(input.shape[0:2], crop_dim)]):
        raise ValueError(f"Arg 'crop_dim'={crop_dim} is too large for 'input' dimensions of {input.shape}. Please choose a smaller crop_dim")

    raw_h, raw_w, _ = input.shape

    # get the number of grids to compute
    num_grid_h = raw_h // crop_dim[0]
    num_grid_w = raw_w // crop_dim[1]

    # crop image
    crop_h, crop_w = num_grid_h * crop_dim[0], num_grid_w * crop_dim[1]

    h_margin = np.floor((raw_h - crop_h)/2).astype(int)
    w_margin = np.floor((raw_w - crop_w)/2).astype(int)

    crop_img = input[h_margin:crop_h+h_margin, w_margin:crop_w+w_margin]

    # calculate strides and make grids
    strides = crop_img.strides
    grid_h_stride = crop_dim[0] * strides[0]
    grid_w_stride = crop_dim[1] * strides[1]

    strided = np.lib.stride_tricks.as_strided(
        crop_img,
        shape=(num_grid_h, num_grid_w, crop_dim[0], crop_dim[1], 3),
        strides=(grid_h_stride, grid_w_stride, strides[0], strides[1], strides[2]),
        writeable=False
        )
    
    return strided, (h_margin, w_margin)

def map_labels_to_target(img_id: str, root_dir: str, dataset_map: dict, img_size: tuple[int, int]=(512, 512)) -> np.ndarray:
    """_summary_

    Args:
        img_id (str): An image id to read in, should not contain file extensions
        root_dir (str): Path to the root directory, ie to images_2021/val or images_2024/train
        dataset_map (dict): A dataset containing the pixel mapping for the dataset

    Raises:
        ValueError: If the path to one of the label masks does not exist.

    Returns:
        np.ndarray: Returns a np.ndarray (dtype=np.uint8) of shape img_size. 
    """
    img_id += ".png"
    mask = np.zeros(shape=(512, 512), dtype=np.uint8)
    for i, name in enumerate(dataset_map['names']):
        if i==0: # skip background class
            continue
        path = Path(root_dir) / "labels" / name / img_id
        if os.path.exists(path):
            label = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if np.any(label): # only grab masks with a non-zero value
                idx = np.where(label != 0)
                label[idx] = dataset_map['mask_vals'][i] # map to correct pixel value
                mask = cv2.bitwise_or(mask, label)
        else:
            raise ValueError(f"The path to {path} does not exist. Please check the funtion arguments")

    return mask

def stack_rgbnir(img_id: str, root_dir: str) -> np.ndarray:
    """_summary_

    Args:
        img_id (str): An image id to read in, should not contain file extensions
        root_dir (str): Path to the root directory, ie to images_2021/val or images_2024/train

    Raises:
        ValueError: If one of the paths to the images does not exist.

    Returns:
        np.ndarray: Returns a np.ndarray (dtype=np.uint8) of shape (H, W, C) with channel order as RGBNIR and C=4.
    """

    img_id += ".jpg"
    nir_path = Path(root_dir) / "images" / "nir" / img_id
    rgb_path = Path(root_dir) / "images" / "rgb" / img_id
    
    if os.path.exists(nir_path) and os.path.exists(rgb_path):
        nir = cv2.imread(str(nir_path), cv2.IMREAD_GRAYSCALE)[..., np.newaxis]
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        stacked = np.concatenate((rgb, nir), axis=2)

        return stacked
    else:
        raise ValueError(f"Error reading the paths for {nir_path} and {rgb_path}. Please check the root_dir {root_dir} and the img_id {img_id}")


if __name__=='__main__':
    img_id = "1AD76MIZN_659-8394-1171-8906"
    stacked = stack_nirrgb(img_id, root_dir='./data/images_2021/train')
    print(stacked)
    print(stacked.shape)

    # import sys
    # print(Path(__file__).parents[2])
    # sys.path.append(Path(__file__).parents[2] / 'data')
    # from data.dataset_maps import class_mapping

    class_mapping = {
    "names": [
        "background",
        "double_plant",
        "drydown",
        "endrow",
        "nutrient_deficiency",
        "planter_skip",
        "water",
        "waterway",
        "weed_cluster"
    ],
    "int_labs": [i for i in range(9)],
    "mask_vals": [0, 50, 75, 100, 125, 150, 175, 200, 255]
}

    map_labels_to_target(img_id, './data/images_2021/train', dataset_map=class_mapping)