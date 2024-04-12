import os
import random
import itertools
from functools import partial
from typing import Optional, Callable, List

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from glob import glob

# from datasets.datasets import *
from src.utils.preprocessing import stack_rgbnir, map_labels_to_target
from src.utils.transforms import null_tfms
from data.dataset_maps import class_mapping

def get_datasets(train_l_dir: str, train_u_dir: str, val_dir: str, test_dir:str , transform_dict: dict, ssl: bool=False):
    """Generates all of the datasets necessary for training.
    If arg 'ssl' is True, then it will generate labeled_train, unlabeled_train, val, and test.
    Otherwise the function will only return train, val, and test
    """
    strong_tfms = transform_dict['strong']
    weak_tfms = transform_dict['weak']
    val_tfms = transform_dict['val']
    test_tfms = transform_dict['test']
    if ssl:
        train_u_ds = AgDataset(root_dir=train_u_dir, ssl_transforms=(weak_tfms(), strong_tfms()))
    else:
        train_u_ds = None
    
    train_l_ds = AgDataset(root_dir=train_l_dir, transforms=strong_tfms())
    val_ds = AgDataset(root_dir=val_dir, transforms=val_tfms())
    # test_ds = AgDataset(root_dir=test_dir, transforms=test_tfms())
    test_ds = None

    return {
        "train": {
            "labeled": train_l_ds,
            "unlabeled": train_u_ds
        },
        "val": val_ds,
        "test": test_ds
    }

class AgDataset(Dataset):

    def __init__(
            self,
            root_dir=None,
            ssl_transforms: tuple[Callable, Callable]=None,
            transforms: Callable=None
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        extensions = ['*.jpg', '*.png']

        if not os.path.exists(self.root_dir):
            raise NotADirectoryError(f'Path to root dir {self.root_dir} does not exist. Please check path integrity')

        self.nir_names = list(itertools.chain.from_iterable(glob(f'{ext}', root_dir=os.path.join(root_dir, "images", "nir")) for ext in extensions))
        self.rgb_names = list(itertools.chain.from_iterable(glob(f'{ext}', root_dir=os.path.join(root_dir, "images", "rgb")) for ext in extensions))

        if len(self.nir_names)==0 or len(self.rgb_names)==0:
            raise FileNotFoundError(f"No images found for root dir {root_dir}. Please check path integrity in the config file and try again.")

        self.ssl_transforms = ssl_transforms
        self.transforms = transforms
        self.null_transform = null_tfms

        print(len(self.nir_names))
        print(len(self.rgb_names))


    def __getitem__(self, index):
        
        # Ensure that the indices don't get out of range with the dataset sampler
        # i.e. force them to loop back around
        index = index % len(self.rgb_names)

        img_id = self.rgb_names[index][:-4]
        stacked = stack_rgbnir(img_id=img_id, root_dir=self.root_dir)

        # apply weak and strong transformations
        if self.ssl_transforms:
            weak_transforms, strong_transforms = self.ssl_transforms
            weak_img_tensor = weak_transforms(image=stacked)['image']

            # convert back to numpy for strong transform of weak image
            weak_img_numpy = np.moveaxis(weak_img_tensor.cpu().numpy(), source=0, destination=2)
            strong_img_tensor = strong_transforms(image=weak_img_numpy)['image']

            return weak_img_tensor, strong_img_tensor
        
        # Return a one set of transformations
        elif self.transforms: 
            mask = map_labels_to_target(img_id=img_id, root_dir=self.root_dir, dataset_map=class_mapping)
            transformed = self.transforms(image=stacked, mask=mask)
            trans_img = transformed["image"]
            trans_mask = transformed["mask"]

            return trans_img, trans_mask
        
        # Else just return a tensor image and mask
        else: 
            mask = map_labels_to_target(img_id=img_id, root_dir=self.root_dir, dataset_map=class_mapping)
            transformed = self.null_transform(image=stacked, mask=mask)
            trans_img = transformed['image']
            trans_mask = transformed['mask']

            return trans_img, trans_mask
    
    def __len__(self):
        return len(self.rgb_names)
