"""
Team Agrivision
This script preprocesses all of the raw tiff files for the 2024 Agriculture Vision Dataset
It removes any null values, clips the pixels from the 1st to the 99th percentile, and normalizes the channels to [0, 255]
Any corrupted images will be skipped.
"""

import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
import os
from argparse import ArgumentParser
from src.utils.preprocessing import normalize_img

parser = ArgumentParser("Preprocess the 2024 Agrivision Dataset")

parser.add_argument('directory', type=str, help='The root directory containing the image folders')
parser.add_argument('--nir', type=str, help='The output directory for the nir images')
parser.add_argument('--rgb', type=str, help='The output directory for the RGB images.')

args = parser.parse_args()

def main():

    folders = glob("*/imagery/", root_dir=args.directory)
    img_names = [str.split(s, " Wu Jing")[0] for s in folders]
    glob_list = [os.path.join(args.directory, f) for f in folders]

    p_bar = tqdm(glob_list, total=len(glob_list))
    for i, dir in tqdm(enumerate(p_bar)):
        p_bar.set_description(desc=f"Processing image {img_names[i]}")
        img_paths = glob('*.tif', root_dir=dir)
    
        stacked = []
        for p in img_paths:
            try:
                channel = cv2.imread(os.path.join(dir, p), -1)
            except cv2.error as e:
                print(e)
                break

            stacked.append(channel)
        norm_bgr, norm_nir = normalize_img(stacked)
        rgb_path = os.path.join(args.rgb, f'{img_names[i]}_rgb.jpg')
        nir_path = os.path.join(args.nir, f'{img_names[i]}_nir.jpg')
        try:
            cv2.imwrite(rgb_path, norm_bgr)
            cv2.imwrite(nir_path, norm_nir)
        except cv2.error as e:
            print(e)
            continue

if __name__ == '__main__':
    main()
    # 1BKXAGHHM