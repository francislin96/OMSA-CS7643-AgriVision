from PIL import Image, ImageEnhance
import cv2
import numpy as np
import torchvision.transforms as standard_transforms
import yaml
import torch
import random
import os
import argparse
from data.dataset_maps import *
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

from sklearn.model_selection import train_test_split, KFold

IMG = 'images'
GT = 'gt'
IDS = 'IDs'

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

def img_basename(filename):
    return os.path.basename(os.path.splitext(filename)[0])

def prepare_gt(root_folder, out_path='gt'):
    if not os.path.exists(os.path.join(root_folder, out_path)):
    # if True:
        print('----------creating groundtruth data for training./.val---------------')
        check_mkdir(os.path.join(root_folder, out_path))
        basname = [img_basename(f) for f in os.listdir(os.path.join(root_folder,'images/rgb'))]
        gt = basname[0]+'.png'
        for fname in basname:
            gtz = np.zeros((512, 512), dtype=int)
            for key in labels_folder.keys():
                gt = fname + '.png'
                #print(os.path.join(root_folder, 'labels', key, gt))
                mask = np.array(cv2.imread(os.path.join(root_folder, 'labels', key, gt), -1)/255, dtype=int) * labels_folder[key]
                gtz[gtz < 1] = mask[gtz < 1]
    
            for key in ['boundaries', 'masks']:
                mask = np.array(cv2.imread(os.path.join(root_folder, key, gt), -1) / 255, dtype=int)
                gtz[mask == 0] = 255
            cv2.imwrite(os.path.join(root_folder, out_path, gt), gtz)


def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)

def scale(img, scale, interpolation=cv2.INTER_LINEAR):
    height, width = img.shape[:2]
    new_height, new_width = int(height * scale), int(width * scale)
    img = cv2.resize(img, (new_width, new_height), interpolation=interpolation)
    return img


def imload(filename, gray=False, scale_rate=1.0, enhance=False):
    if not gray:
        image = cv2.imread(filename)  # cv2 read color image as BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # (h, w, 3)
        if scale_rate != 1.0:
            image = scale(image, scale_rate)
        if enhance:
            image = Image.fromarray(np.asarray(image, dtype='uint8'))
            contrast = ImageEnhance.Contrast(image)
            image = contrast.enhance(1.55)
    else:
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)  # read gray image
       
        if scale_rate != 1.0:
            image = scale(image, scale_rate, interpolation=cv2.INTER_NEAREST)
        image = np.asarray(image, dtype='uint8')

    return image


def img_mask_crop(image, mask, size=(256, 256), limits=(224, 512)):
    rc = RandomSizedCrop(height=size[0], width=size[1], min_max_height=limits)
    crops = rc(image=image, mask=mask)
    return crops['image'], crops['mask']


def img_mask_pad(image, mask, target=(288, 288)):
    padding = PadIfNeeded(p=1.0, min_height=target[0], min_width=target[1])
    paded = padding(image=image, mask=mask)
    return paded['image'], paded['mask']


# def composed_augmentation(image, mask):
#     aug = Compose([
#         VerticalFlip(p=0.5),
#         HorizontalFlip(p=0.5),
#         RandomRotate90(p=0.5),
#         HueSaturationValue(hue_shift_limit=20,
#                            sat_shift_limit=5,
#                            val_shift_limit=15, p=0.5),

#         OneOf([
#             GridDistortion(p=0.5),
#             Transpose(p=0.5)
#         ], p=0.5),

#         CLAHE(p=0.5)
#     ])

#     auged = aug(image=image, mask=mask)
#     return auged['image'], auged['mask']


# def get_random_pos(img, window_shape):
#     """ Extract of 2D random patch of shape window_shape in the image """
#     w, h = window_shape
#     W, H = img.shape[-2:]
#     x1 = random.randint(0, W - w - 1)
#     x2 = x1 + w
#     y1 = random.randint(0, H - h - 1)
#     y2 = y1 + h
#     return x1, x2, y1, y2


def get_training_list(root_folder, count_label=True):
    dict_list = {}
    basename = [img_basename(f) for f in os.listdir(os.path.join(root_folder, 'images/nir'))]
    if count_label:
        for key in labels_folder.keys():
            no_zero_files=[]
            for fname in basename:
                gt = np.array(cv2.imread(os.path.join(root_folder, 'labels', key, fname+'.png'), -1))
                if np.count_nonzero(gt):
                    no_zero_files.append(fname)
                else:
                    continue
            dict_list[key] = no_zero_files
    return dict_list, basename

def get_unlabeled_list(root_folder):
    dict_list = {}
    basename = [img_basename(f) for f in os.listdir(os.path.join(root_folder, 'images/nir'))]
    return dict_list, basename


def split_train_val_test_sets(args,data_folder=Data_Folder, name='Agriculture', bands=['NIR','RGB'], KF=3, k=1, seeds=69278):

    TRAIN_ROOT = os.path.join(args.dataset_root, 'train')
    VAL_ROOT = os.path.join(args.dataset_root, 'val')
    TEST_ROOT = os.path.join(args.dataset_root, 'test')

    train_id, t_list = get_training_list(root_folder=TRAIN_ROOT, count_label=False)
    val_id, v_list = get_training_list(root_folder=VAL_ROOT, count_label=False)
    test_id, test_list = get_training_list(root_folder=TEST_ROOT, count_label=False)
    
    if KF >=2:
        kf = KFold(n_splits=KF, shuffle=True, random_state=seeds)
        val_ids = np.array(v_list)
        idx = list(kf.split(np.array(val_ids)))
        if k >= KF:  # k should not be out of KF range, otherwise set k = 0
            k = 0
        t2_list, v_list = list(val_ids[idx[k][0]]), list(val_ids[idx[k][1]])
    else:
        t2_list=[]

    img_folders = [os.path.join(data_folder[name]['ROOT'], 'train', data_folder[name][band]) for band in bands]
    gt_folder = os.path.join(data_folder[name]['ROOT'], 'train', data_folder[name]['GT'])

    val_folders = [os.path.join(data_folder[name]['ROOT'], 'val', data_folder[name][band]) for band in bands]
    val_gt_folder = os.path.join(data_folder[name]['ROOT'], 'val', data_folder[name]['GT'])

    test_folders = [os.path.join(data_folder[name]['ROOT'], 'test', data_folder[name][band]) for band in bands]
    test_gt_folder = os.path.join(data_folder[name]['ROOT'], 'test', data_folder[name]['GT'])
    
   
    train_dict = {
        IDS: train_id,
        IMG: [[img_folder.format(id) for img_folder in img_folders] for id in t_list] +
             [[val_folder.format(id) for val_folder in val_folders] for id in t2_list],
        GT: [gt_folder.format(id) for id in t_list] + [val_gt_folder.format(id) for id in t2_list],
        'all_files': t_list + t2_list
    }

    val_dict = {
        IDS: val_id,
        IMG: [[val_folder.format(id) for val_folder in val_folders] for id in v_list],
        GT: [val_gt_folder.format(id) for id in v_list],
        'all_files': v_list
    }

    test_dict = {
        IDS: test_id,
        IMG: [[test_folder.format(id) for test_folder in test_folders] for id in test_list],
        GT: [test_gt_folder.format(id) for id in test_list],
        'all_files': test_list
    }
    
    print('train set -------', len(train_dict[GT]))
    print('val set ---------', len(val_dict[GT]))
    print('test set ---------', len(test_dict[GT]))
    return train_dict, val_dict, test_dict

def get_unlabeled_set(args, data_folder=Data_Folder, name="Agriculture", bands=['NIR','RGB'], KF=3, k=1, seeds=69278):
    UNLABELED_ROOT = "./data/images_2024/train"
    unlabeled_ids, u_list = get_unlabeled_list(root_folder=UNLABELED_ROOT)
    unlabeled_folders = [os.path.join(data_folder[name]['UNLABELED_ROOT'], 'train', data_folder[name][band]) for band in bands]
    unlabeled_dict = {
    IDS: unlabeled_ids,
    IMG: [[unlabeled_folder.format(id) for unlabeled_folder in unlabeled_folders] for id in u_list],
    'all_files': u_list}
    print('unlabeled set -------', len(unlabeled_dict[IMG]))
    return unlabeled_dict


def load_yaml_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)
        
def set_args_attr(config, args):
    for k, v in config.items():
        if type(v) is dict:
            set_args_attr(v, args)
        elif getattr(args, k, None) is None:
            setattr(args, k, v)
    return args



def random_seed(seed_value, use_cuda=True):
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False



mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def get_visualize(args):
    visualize = standard_transforms.Compose([
        standard_transforms.Resize(300),
        standard_transforms.CenterCrop(300),
        standard_transforms.ToTensor()
    ])

    if args.pre_norm:
        restore = standard_transforms.Compose([
            DeNormalize(*mean_std),
            standard_transforms.ToPILImage(),
        ])
    else:
        restore = standard_transforms.Compose([
            standard_transforms.ToPILImage(),
        ])

    return visualize, restore


def setup_palette(palette):
    palette_rgb = []
    for _, color in palette.items():
        palette_rgb += color

    zero_pad = 256 * 3 - len(palette_rgb)

    for i in range(zero_pad):
        palette_rgb.append(0)

    return palette_rgb


def colorize_mask(mask, palette):
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(setup_palette(palette))

    return new_mask


def convert_to_color(arr_2d, palette):
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d
