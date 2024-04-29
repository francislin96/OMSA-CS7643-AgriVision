import torch
import argparse
import yaml
import json
from src.train import train
from src.datasets.fixmatch_datasets import get_datasets
from src.datasets.dataloaders import get_dataloaders
from src.models.create_models import deeplabv3_plus, fpn, unet_plusplus
from src.utils.transforms import train_tfms, strong_tfms, weak_tfms, null_tfms
from src.loss.criterions import DiceLoss, TverskyLoss, FocalTverskyLoss, ACW_loss
from torch.nn import CrossEntropyLoss


def main(args):

    transform_dict = {
        'train': train_tfms,
        'val': null_tfms,
        'test': null_tfms
    }

    ds_dict = get_datasets(
        train_l_dir=args.train_l_dir, 
        val_dir=args.val_dir, 
        test_dir=args.test_dir, 
        transform_dict=transform_dict, 
        ssl=args.ssl
    )

    train_l_ds = ds_dict['train']['labeled']
    val_ds = ds_dict['val']
    test_ds = ds_dict['test']

    (train_l_loader, _), val_loader, test_loader = get_dataloaders(
        train_l_ds=train_l_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        batch_size=args.batch_size
    )

    # Instantiate model
    if args.model=='deeplab':
        model = deeplabv3_plus(args).to(args.device)
    elif args.model=='unet_plusplus':
        model = unet_plusplus(args).to(args.device)
    elif args.model=='fpn':
        model = fpn(args).to(args.device)
    else:
        raise ValueError("Please select a valid model type")
    
    # Get loss function
    if args.loss_fn=='dice':
        criterion = DiceLoss(args)
    elif args.loss_fn=='tversky':
        criterion = TverskyLoss(args)
    elif args.loss_fn=='focal_tversky':
        criterion = FocalTverskyLoss(args)
    elif args.loss_fn=='acw':
        criterion = ACW_loss(args)
    else:
        print("Using cross entropy loss")
        criterion = CrossEntropyLoss(reduce='mean')

    model, epoch_meters = train(args, model, train_l_loader, val_loader, criterion, filter_bias_and_bn=True)

    with open(f'{args.model_run_name}_epoch_meters.json', 'w') as f:
        json.dump(epoch_meters, f)

if __name__ == '__main__':

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
        
    parser = argparse.ArgumentParser(description='Train a segmentation model using Fixmatch SSL')
    parser.add_argument('config', type=str, help='Path to the run yaml configuration file')
    args = parser.parse_args()
    print(args.config)

    config = load_yaml_config(str(args.config))

    args = set_args_attr(config, args)

    main(args)