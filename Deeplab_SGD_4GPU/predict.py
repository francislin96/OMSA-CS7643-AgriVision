import os
import random
import cv2
import numpy as np
from tqdm import tqdm
from collections import Counter, OrderedDict
import time
import matplotlib.pyplot as plt
import torch
import datetime
from torch.utils.data import DataLoader
import torchvision.transforms as standard_transforms
from torch import Tensor

import torchvision.utils as vutils
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter

import yaml
import argparse


from typing import Callable, Tuple

from functools import partial
from src.utils.funcs import *
from data.dataset_maps import class_mapping
from src.datasets.datasets import AgricultureDataset, agriculture_configs
from src.loss.criterions import ACW_loss
from src.models.create_models import deeplabv3_plus
from src.metrics import *


def test(net, test_loader, criterion):
    net.eval()
    gts_all, predictions_all = [], []
    with torch.no_grad():
        for ti, (inputs, gts) in enumerate(test_loader):
            inputs, gts = inputs.cuda(), gts.cuda()
            outputs = net(inputs)
            gts_all.append(gts.data.squeeze(0).cpu().numpy())
            predictions = outputs.data.max(1)[1].squeeze(1).squeeze(0).cpu().numpy()
            predictions_all.append(predictions)

    return gts_all, predictions_all

def main(args):

    TRAIN_ROOT = os.path.join(args.dataset_root,'train')
    VAL_ROOT = os.path.join(args.dataset_root,'val')
    TEST_ROOT = os.path.join(args.dataset_root,'test')
    UNLABELED_ROOT = "./data/images_2024/train"
    
    
    train_args = agriculture_configs(args, net_name='DeepLabV3plus',
                                 data='Agriculture',
                                 bands_list=['NIR', 'RGB'],
                                 kf=0, k_folder=0,
                                 note='DeepLabTraining'
                                 )

       
    criterion = ACW_loss(args).cuda()
    model = deeplabv3_plus(args)
    
    prepare_gt(TEST_ROOT)
    cpt_path = '/Users/sshel/OneDrive/Documents/OMSA/CS 7643/ckpt/plugged_in_deeplab_SGD/epoch_19_loss_0.45038_acc_0.82621_acc-cls_0.74087_mean-iu_0.60962_fwavacc_0.70690_f1_0.74885_lr_0.0000671535.pth'
    checkpoint = torch.load(cpt_path)
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:]
        new_state_dict[name]=v
    model.load_state_dict(new_state_dict)
    model.cuda()
    _, _,test_set, _,_ = train_args.get_dataset(args)
    test_loader = DataLoader(dataset=test_set, batch_size=1)
    gts, predictions = test(model, test_loader, criterion)
    acc, acc_cls, mean_iu, fwavacc, f1, iu = evaluate(predictions, gts, 9)
    print(acc, acc_cls, mean_iu, f1)
    for i, u in enumerate(iu):
        print("IoU for {} class: {:.3f}".format(class_mapping['names'][i],iu[i]))

if __name__ == '__main__':

        
    parser = argparse.ArgumentParser(description='Train a segmentation model using SSL')
    parser.add_argument('-config', type=str, help='Path to the run yaml configuration file', required=True)
    parser.add_argument('--dataset_root', type=str, default="./data/images_2021", help="Path to dataset")
    parser.add_argument('--node_size', type=tuple, default=(32,32), help="MSCG model node size")
    parser.add_argument('--best_record', type=dict, default={})
    parser.add_argument('-loader', default=AgricultureDataset, help="Dataset")
    args = parser.parse_args()
    config = load_yaml_config(str(args.config))
    args = set_args_attr(config, args)
    main(args)