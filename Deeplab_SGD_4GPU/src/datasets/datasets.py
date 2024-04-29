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
from torch.utils.data import Dataset
import torchvision.transforms as standard_transforms
from albumentations import (
    Compose,
    OneOf,
    Flip,
    PadIfNeeded,
    GaussNoise,
    MotionBlur,
    OpticalDistortion,
    RandomSizedCrop,
    HorizontalFlip,
    VerticalFlip,
    RandomRotate90,
    ShiftScaleRotate,
    CenterCrop,
    Transpose,
    GridDistortion,
    ElasticTransform,
    RandomGamma,
    RandomBrightnessContrast,
    CLAHE,
    HueSaturationValue,
    Blur,
    MedianBlur,
    ChannelShuffle
  
)
from albumentations.pytorch.transforms import ToTensorV2

import torch
from src.utils.funcs import *
from torch import Tensor



def get_datasets(train_l_dir: str, val_dir: str, test_dir:str, transform_dict: dict, train_u_dir: str=None, ssl: bool=False):
    """Generates all of the datasets necessary for training.
    If arg 'ssl' is True, then it will generate labeled_train, unlabeled_train, val, and test.
    Otherwise the function will only return train, val, and test
    """

    train_tfms = transform_dict['train']
    val_tfms = transform_dict['val']
    test_tfms = transform_dict['test']

    if ssl:
        strong_tfms = transform_dict['strong']
        weak_tfms = transform_dict['weak']
        train_u_ds = AgDataset(root_dir=train_u_dir, ssl_transforms=(weak_tfms(), strong_tfms()))
    else:
        train_u_ds = None
    
    train_l_ds = AgDataset(root_dir=train_l_dir, transforms=train_tfms())
    val_ds = AgDataset(root_dir=val_dir, transforms=val_tfms())
    test_ds = AgDataset(root_dir=test_dir, transforms=test_tfms())

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
        
        if len(self.nir_names) != len(self.rgb_names):
            raise ValueError(f"Mismatch in the number of NIR file names: {len(self.nir_names)} and RGB file names: {len(self.rgb_names)}")

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
        stacked, mask = stack_rgbnir(img_id=img_id, root_dir=self.root_dir)

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
            target = map_labels_to_target(img_id=img_id, root_dir=self.root_dir, dataset_map=class_mapping)

            transformed = self.transforms(image=stacked, target=target, mask=mask)
            trans_img = transformed["image"]
            trans_target = transformed["target"]
            trans_mask = transformed["mask"]

            trans_img *= trans_mask
            trans_target *= trans_mask

            return trans_img, trans_target

        # Else just return a tensor image and mask
        else: 
            target = map_labels_to_target(img_id=img_id, root_dir=self.root_dir, dataset_map=class_mapping)

            transformed = self.null_transform(image=stacked, target=target, mask=mask)
            trans_img = transformed["image"]
            trans_target = transformed["target"]
            trans_mask = transformed["mask"]

            trans_img *= trans_mask
            trans_target *= trans_mask

            return trans_img, trans_target
    
    def __len__(self):
        return len(self.rgb_names)

### Heavily adapted from https://github.com/samleoqh/MSCG-Net   
class AgricultureDataset(Dataset):
    def __init__(self, mode='train', file_lists=None, windSize=(256, 256),
                 num_samples=10000, pre_norm=False, scale=1.0 / 1.0):
        assert mode in ['train', 'val', 'test','unlabeled', 'semisupervised']
        self.mode = mode
        self.norm = pre_norm
        self.winsize = windSize
        self.samples = num_samples
        self.scale = scale
        self.all_ids = file_lists['all_files']
    
        self.image_files = file_lists[IMG]
        if self.mode in ['train','val','test']:
            self.mask_files = file_lists[GT]
        elif self.mode == 'semisupervised':
            self.mask_files = torch.load("pseudo_labels_tensor.pt").numpy()
            non_zero_mask = np.any(self.mask_files != 0, axis=(1, 2))
            self.mask_files = self.mask_files[non_zero_mask]
            indices_to_keep = [i for i, mask in enumerate(self.mask_files) if np.any(mask != 0)]
            self.all_ids = [self.all_ids[i] for i in indices_to_keep]
                    
        
    def __len__(self):
        return len(self.all_ids)

    def __getitem__(self, idx):
        
        if len(self.image_files) > 1:
            
            imgs = []
            
            for k in range(len(self.image_files[idx])):
                
                filename = self.image_files[idx][k]
                path, _ = os.path.split(filename)
                
                if "nir" in path:
                    img = imload(filename, gray=True, scale_rate=self.scale)
                    img = np.expand_dims(img, 2)
                    
                    imgs.append(img)
                    
                else:
                    img = imload(filename, scale_rate=self.scale)
                    imgs.append(img)
            image = np.concatenate(imgs, 2)
            
        else:
            
            filename = self.image_files[idx][0]
            path, _ = os.path.split(filename)
            if 'nir' in path:
                image = imload(filename, gray=True, scale_rate=self.scale)
                image = np.expand_dims(image, 2)
            else:
                image = imload(filename, scale_rate=self.scale)
        if self.mode == 'unlabeled':
            label = np.random.randint(0, 256, size=(512, 512), dtype=np.uint8)
        elif self.mode == 'semisupervised':
            
            label = self.mask_files[idx]
            # print(np.all(label==0))        
        else:
            label = imload(self.mask_files[idx], gray=True, scale_rate=self.scale)

        # if self.winsize != label.shape:
        #     image, label = img_mask_crop(image=image, mask=label,
        #                                  size=self.winsize, limits=self.winsize)

        if self.mode == 'train':
            image_p, label_p = self.train_augmentation(image, label)
        elif self.mode == 'val':
            image_p, label_p = self.val_augmentation(image, label)
        
        elif self.mode == 'test':
            image_p, label_p = self.test_augmentation(image, label)

        elif self.mode == 'unlabeled':
            image_p, label_p = self.unlabeled_augmentation(image,label)
        
        elif self.mode == 'semisupervised':
            image_p, label_p = self.train_augmentation(image, label)
            
        image_p = np.asarray(image_p, np.float32).transpose((2, 0, 1)) / 255.0
        label_p = np.asarray(label_p, dtype='int64')

        image_p, label_p = torch.from_numpy(image_p), torch.from_numpy(label_p)

        if self.norm:
            image_p = self.normalize(image_p)

        return image_p, label_p

    @classmethod
    def train_augmentation(cls, img, mask):
        aug = Compose([
            VerticalFlip(p=0.5),
            HorizontalFlip(p=0.5),
            RandomRotate90(p=0.5),
            #MedianBlur(p=0.2),
            # Transpose(p=0.5),
            # RandomSizedCrop(min_max_height=(128, 512), height=512, width=512, p=0.1),
            # ShiftScaleRotate(p=0.2,
            #                  rotate_limit=10, scale_limit=0.1),
            # ChannelShuffle(p=0.1),
        ])

        auged = aug(image=img, mask=mask)
        return auged['image'], auged['mask']

    @classmethod
    def val_augmentation(cls, img, mask):
        aug = Compose([
            VerticalFlip(p=0.5),
            HorizontalFlip(p=0.5),
            RandomRotate90(p=0.5),
        ])

        auged = aug(image=img, mask=mask)
        return auged['image'], auged['mask']
    
    @classmethod
    def test_augmentation(cls, img, mask):
        # aug = Compose([
        
        #     ToTensorV2(p=1.0)
        # ])
        # auged = aug(image=img, mask=mask)
        # return auged['image'], auged['mask']
        return torch.tensor(img), torch.tensor(mask)
    
    @classmethod
    def unlabeled_augmentation(cls, img, label):
        return torch.tensor(img), torch.tensor(label)
    
    @classmethod
    def normalize(cls, img):
        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        norm = standard_transforms.Compose([standard_transforms.Normalize(*mean_std)])
        return norm(img)


class agriculture_configs(object):
    
    def __init__ (self, args, net_name, data, bands_list, kf=1, k_folder=5, note=''):
        self.model = net_name
        self.dataset = data
        self.bands = bands_list
        self.k = kf
        self.k_folder = k_folder
        self.suffix_note = note
        self.loader = args.loader
        self.seeds = args.seeds
        self.pre_norm = args.pre_norm
        self.input_size = args.input_size
        self.scale_rate = args.scale_rate
        self.val_samples = args.val_samples
        self.train_samples = args.train_samples
        self.val_size = args.val_size
        self.snapshot = args.snapshot

        check_mkdir(args.ckpt_path)
        check_mkdir(os.path.join(args.ckpt_path, self.model))

        bandstr = '-'.join(self.bands)
        if self.k_folder is not None:
            subfolder = self.dataset + '_' + bandstr + '_kf-' + str(self.k_folder) + '-' + str(self.k)
        else:
            subfolder = self.dataset + '_' + bandstr
        if note != '':
            subfolder += '-'
            subfolder += note

        check_mkdir(os.path.join(args.ckpt_path, self.model, subfolder))
        self.save_path = os.path.join(args.ckpt_path, self.model, subfolder)
    
    def get_unlabeled_list(self,args):
        return get_unlabeled_set(args, name=self.dataset, bands=self.bands, KF=self.k_folder, k=self.k, seeds=self.seeds)
    
    def get_file_list(self,args):
        return split_train_val_test_sets(args, name=self.dataset, bands=self.bands, KF=self.k_folder, k=self.k, seeds=self.seeds)

    def get_dataset(self,args, semisupervised=False):
        train_dict, val_dict, test_dict = self.get_file_list(args)
        unlabeled_dict = self.get_unlabeled_list(args)
       
        train_set = self.loader(mode='train', file_lists=train_dict, pre_norm=self.pre_norm,
                                    num_samples=self.train_samples, windSize=self.input_size, scale=self.scale_rate)
        val_set = self.loader(mode='val', file_lists=val_dict, pre_norm=self.pre_norm,
                                  num_samples=self.val_samples, windSize=self.val_size, scale=self.scale_rate)
        test_set = self.loader(mode='test', file_lists=test_dict, pre_norm=self.pre_norm,  num_samples=self.val_samples, 
                               windSize=self.val_size, scale=self.scale_rate)
        unlabeled_set = self.loader(mode='unlabeled', file_lists=unlabeled_dict, pre_norm=self.pre_norm, num_samples=self.val_samples,
                                    windSize=self.input_size, scale=self.scale_rate)

        
        if semisupervised:
            semisupervised_set = self.loader(mode='semisupervised', file_lists=unlabeled_dict, pre_norm=self.pre_norm,
                                             num_samples=self.train_samples, windSize=self.input_size, scale=self.scale_rate)        
        else:
            semisupervised_set = None
        return train_set, val_set, test_set, unlabeled_set, semisupervised_set


    def resume_train(self, net):
        if len(self.snapshot) == 0:
            curr_epoch = 1
            self.best_record = {'epoch': 0, 'val_loss': 0, 'acc': 0, 'acc_cls': 0, 'mean_iu': 0, 'fwavacc': 0,
                                'f1': 0}
            print('Training from scratch.')
        else:
            print('training resumes from ' + self.snapshot)
            
            state_dict = torch.load(os.path.join(self.save_path, self.snapshot))
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('module.'):
                    new_key = key[7:]  # Remove 'module.' prefix
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            
            net.load_state_dict(new_state_dict)
            split_snapshot = self.snapshot.split('_')
            curr_epoch = int(split_snapshot[1]) + 1
            self.best_record = {'epoch': int(split_snapshot[1]), 'val_loss': float(split_snapshot[3]),
                                'acc': float(split_snapshot[5]), 'acc_cls': float(split_snapshot[7]),
                                'mean_iu': float(split_snapshot[9]), 'fwavacc': float(split_snapshot[11]),
                                'f1': float(split_snapshot[13])}
        return net, curr_epoch


    def print_best_record(self):
        print(
            '[best_ %d]: [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f], [f1 %.5f]' % (
                self.best_record['epoch'],
                self.best_record['val_loss'], self.best_record['acc'],
                self.best_record['acc_cls'],
                self.best_record['mean_iu'], self.best_record['fwavacc'], self.best_record['f1']
            ))


    def update_best_record(self, epoch, val_loss,
                           acc, acc_cls, mean_iu,
                           fwavacc, f1):
        print('----------------------------------------------------------------------------------------')
        print('[epoch %d]: [val loss %.5f], [acc %.5f], [acc_cls %.5f], [mean_iu %.5f], [fwavacc %.5f], [f1 %.5f]' % (
            epoch, val_loss, acc, acc_cls, mean_iu, fwavacc, f1))
        self.print_best_record()

        print('----------------------------------------------------------------------------------------')
        if mean_iu > self.best_record['mean_iu'] or f1 > self.best_record['f1']:
            self.best_record['epoch'] = epoch
            self.best_record['val_loss'] = val_loss
            self.best_record['acc'] = acc
            self.best_record['acc_cls'] = acc_cls
            self.best_record['mean_iu'] = mean_iu
            self.best_record['fwavacc'] = fwavacc
            self.best_record['f1'] = f1
            return True
        else:
            return False

    def display(self):
        """printout all configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")
