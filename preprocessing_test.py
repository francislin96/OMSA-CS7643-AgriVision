from src.utils.preprocessing import read_and_stack, normalize_img, split_img, apply_boundary_to_img, crop_to_cnt
import cv2
import numpy as np

boundary_path = './data/images_2024/raw_tiff_files/266J6I3NU Wu Jing/boundary.json'
img_dir = './data/images_2024/raw_tiff_files/266J6I3NU Wu Jing/imagery'

stacked = read_and_stack(img_dir)
norm_bgr, norm_nir = normalize_img(stacked, (5, 95))
mask, masked_bgr = apply_boundary_to_img(boundary_path=boundary_path, img=norm_bgr)
_, masked_nir = apply_boundary_to_img(boundary_path=boundary_path, img=norm_nir)


cv2.namedWindow('test', cv2.WINDOW_NORMAL)
cv2.imshow('test', norm_bgr)
cv2.waitKey(0)
cv2.imshow('test', masked_bgr)
cv2.waitKey(0)
cv2.imshow('test', masked_nir)
cv2.waitKey(0)

min_img = crop_to_cnt(mask, masked_bgr)
cv2.imshow('test', min_img)
cv2.waitKey(0)

img_crops, margins = split_img(min_img)
for n in range(img_crops.shape[0]):
    for m in range(img_crops.shape[1]):
        img = img_crops[n, m]
        bin_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if np.sum(bin_img > 0)/(np.product(bin_img.shape)) < 0.75:
            continue
        cv2.imshow('test', img)
        cv2.waitKey(0)
cv2.destroyAllWindows()