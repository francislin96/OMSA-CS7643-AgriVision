# dummy_data.py
# Functions to generate random NIR and RGB images with masks corresponding to the class mapping to iterate over for development

import os
import cv2
import numpy as np
from data.dataset_maps import class_mapping
from src.datasets import dataloaders

def generate_blob(shape, intensity, num_blobs):
    image = np.zeros(shape, dtype=np.uint8)
    for _ in range(num_blobs):
        # Randomly generate the center and radius
        center_x = np.random.randint(0, shape[1])
        center_y = np.random.randint(0, shape[0])
        radius = np.random.randint(10, 50)

        # Draw the blob
        cv2.circle(image, (center_x, center_y), radius, intensity, -1)
    return image

def create_dataset(num_images, rgb_dir, nir_dir, mask_dir=None):
    os.makedirs(rgb_dir, exist_ok=True)
    if mask_dir:
        os.makedirs(mask_dir, exist_ok=True)
    for i in range(num_images):
        # Generate random NIR-RGB image
        nir = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
        rgb = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
        # nirrgb = np.concatenate((nir[..., None], rgb), axis=-1)

        # Generate mask with random blobs for each class
        if mask_dir:
  
            mask = np.zeros((512, 512), dtype=np.uint8)
            for intensity in class_mapping['mask_vals']:
                if intensity != 0:  # Skip background
                    num_blobs = np.random.randint(1, 5)
                    mask += generate_blob(mask.shape, intensity, num_blobs)

            # Apply mask where the intensity is the highest (handle overlapping blobs)
            mask = np.argmax(np.stack([mask == v for v in class_mapping['mask_vals']], axis=-1), axis=-1)
            mask = np.choose(mask, class_mapping['mask_vals'])
            cv2.imwrite(os.path.join(mask_dir, f'mask_{i}.png'), mask)

        # Save the images and masks
        cv2.imwrite(os.path.join(rgb_dir, f'image_{i}.png'), rgb)
        cv2.imwrite(os.path.join(nir_dir,f'image_{i}.png'), nir)
        
