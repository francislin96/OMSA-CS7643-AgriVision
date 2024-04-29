"""
Team Agrivision
This script preprocesses all of the raw tiff files for the 2024 Agriculture Vision Dataset
It removes any null values, clips the pixels from the 1st to the 99th percentile, and normalizes the channels to [0, 255]
Any corrupted images will be skipped.
"""
import logging
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import os
import math
from argparse import ArgumentParser
from src.utils.preprocessing import stack_tifs, normalize_img, crop_to_cnt, apply_boundary_to_img, split_img
import multiprocessing

parser = ArgumentParser("Preprocess the 2024 Agrivision Dataset")

parser.add_argument('input_dir', type=str, help='The root directory containing the input image folders')
parser.add_argument('--nir', type=str, help='The output directory for the nir images')
parser.add_argument('--rgb', type=str, help='The output directory for the RGB images.')

args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG,
                    filename='logs/img_proc_debug.log', 
                    filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger('debug_logger')

def process_directory(dir_and_img_name: tuple):

    dir, img_name = dir_and_img_name
    print(img_name)
    stacked = stack_tifs(dir)
    if stacked is None:
        print(f"Broken tif file at {dir}, skipping")
        logger.error(f"Broken tif file at {dir}, skipping")
        return None
    bgr_img, nir_img = normalize_img(stacked, (1, 99))

    # Apply masks if boundary file and trim excess
    bound_json = dir.split("imagery/")[0]+'boundary.json'
    if os.path.exists(bound_json):
        mask, masked_bgr = apply_boundary_to_img(boundary_path=bound_json, img=bgr_img, img_name=img_name)
        if mask is None:
            print(f"Broken boundary files at {bound_json}, skipping")
            logger.debug(f"Broken boundary files at {bound_json}, skipping")
            return None
        _, masked_nir = apply_boundary_to_img(boundary_path=bound_json, img=nir_img, img_name=img_name)

        bgr_img = crop_to_cnt(mask, masked_bgr)
        nir_img = crop_to_cnt(mask, masked_nir)
        
    bgr_crops, bgr_margins = split_img(bgr_img)
    nir_crops, nir_margins = split_img(nir_img)

    # Loop through cropped images and save when img area > tau
    for n in range(bgr_crops.shape[0]):
        for m in range(bgr_crops.shape[1]):
            bgr_img = bgr_crops[n, m]
            nir_img = nir_crops[n, m]
            if np.sum(nir_img > 0)/(np.prod(nir_img.shape)) < 0.75:
                continue

            rgb_path = os.path.join(args.rgb, "_".join([img_name, f"row{n}", f"col{m}"])+".jpg")
            nir_path = os.path.join(args.nir, "_".join([img_name, f"row{n}", f"col{m}"])+".jpg")
            cv2.imwrite(rgb_path, bgr_img)
            cv2.imwrite(nir_path, nir_img)

def main():

    folders = glob("*/imagery/", root_dir=args.input_dir)
    img_names = [str.split(s, " Wu Jing")[0] for s in folders]
    glob_list = [os.path.join(args.input_dir, f) for f in folders]

    folders = glob("*/imagery/", root_dir=args.input_dir)
    img_names = [str.split(s, " Wu Jing")[0] for s in folders]
    glob_list = [os.path.join(args.input_dir, f) for f in folders]
    dir_and_names = list(zip(glob_list, img_names))


    num_processes = multiprocessing.cpu_count()
    num_processes = 8
    

    with multiprocessing.Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap(process_directory, dir_and_names), total=len(dir_and_names)))
        # list(tqdm(pool.imap(process_directory, dir_and_names)))

    # p_bar = tqdm(glob_list, total=len(glob_list))


    # for i, dir in tqdm(enumerate(p_bar)):
        # p_bar.set_description(desc=f"Processing image {img_names[i]}")

        # process_directory(dir=dir)
        # Read in all channels and normalize
        # stacked = read_and_stack(dir)
        # bgr_img, nir_img = normalize_img(stacked, (1, 99))

        # # Apply masks if boundary file and trim excess
        # bound_json = dir.split("imagery/")[0]+'boundary.json'
        # if os.path.exists(bound_json):
        #     mask, masked_bgr = apply_boundary_to_img(boundary_path=bound_json, img=bgr_img)
        #     _, masked_nir = apply_boundary_to_img(boundary_path=bound_json, img=nir_img)

        #     bgr_img = crop_to_cnt(mask, masked_bgr)
        #     nir_img = crop_to_cnt(mask, masked_nir)
        #     print(nir_img.shape)
        
        # bgr_crops, bgr_margins = split_img(bgr_img)
        # nir_crops, nir_margins = split_img(nir_img)

        # # Loop through cropped images and save when img area > tau
        # for n in range(bgr_crops.shape[0]):
        #     for m in range(bgr_crops.shape[1]):
        #         bgr_img = bgr_crops[n, m]
        #         nir_img = nir_crops[n, m]
        #         if np.sum(nir_img > 0)/(np.prod(nir_img.shape)) < 0.75:
        #             continue

        #         rgb_path = os.path.join(args.rgb, "_".join([img_names[i], f"row{n}", f"col{m}"])+".jpg")
        #         nir_path = os.path.join(args.nir, "_".join([img_names[i], f"row{n}", f"col{m}"])+".jpg")
        #         cv2.imwrite(rgb_path, bgr_img)
        #         cv2.imwrite(nir_path, nir_img)


        # print(stacked)
        # img_paths = glob('*.tif', root_dir=dir)
    
        # stacked = []
        # for p in img_paths:
        #     try:
        #         channel = cv2.imread(os.path.join(dir, p), -1)
        #     except cv2.error as e:
        #         print(e)
        #         break

        #     stacked.append(channel)
        # norm_bgr, norm_nir = normalize_img(stacked)
        # rgb_path = os.path.join(args.rgb, f'{img_names[i]}_rgb.jpg')
        # nir_path = os.path.join(args.nir, f'{img_names[i]}_nir.jpg')
        # try:
        #     cv2.imwrite(rgb_path, norm_bgr)
        #     cv2.imwrite(nir_path, norm_nir)
        # except cv2.error as e:
        #     print(e)
        #     continue

if __name__ == '__main__':
    main()