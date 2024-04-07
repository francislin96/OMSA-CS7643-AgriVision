# generate_dev_data.py
# Use this script to generate a bunch of fake RGB and NIR images with masks
# that correspond to the class mappings

from src.utils.dummy_data import generate_blob, create_dataset

# Labeled Dataset
#train
create_dataset(
    num_images=50, 
    rgb_dir='data/dev_data/labeled/train/rgb',
    nir_dir='data/dev_data/labeled/train/nir',
    mask_dir='data/dev_data/labeled/train/masks'
    )

#val
create_dataset(
    num_images=20, 
    rgb_dir='data/dev_data/labeled/val/rgb',
    nir_dir='data/dev_data/labeled/val/nir',
    mask_dir='data/dev_data/labeled/val/masks'
    )

#test
create_dataset(
    num_images=10, 
    rgb_dir='data/dev_data/labeled/test/rgb',
    nir_dir='data/dev_data/labeled/test/nir',
    mask_dir='data/dev_data/labeled/test/masks'
    )

# Unlabled Dataset
create_dataset(
    num_images=30, 
    rgb_dir='data/dev_data/unlabeled/rgb',
    nir_dir='data/dev_data/unlabeled/nir',
    )