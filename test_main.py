import torch
import argparse
import yaml
from src.train import train
from src.datasets.fixmatch_datasets import get_datasets
from src.datasets.dataloaders import get_dataloaders
from src.models.create_models import deeplabv3_plus
from src.utils.transforms import strong_tfms, weak_tfms, null_tfms


def main(args):

    transform_dict = {
        'strong': strong_tfms,
        'weak': weak_tfms,
        'val': null_tfms,
        'test': null_tfms
    }

    ds_dict = get_datasets(
        train_l_dir='./data/dev_data/labeled/train', 
        train_u_dir='./data/dev_data/unlabeled/train', 
        val_dir='./data/dev_data/labeled/val', 
        test_dir='./data/dev_data/labeled/test', 
        transform_dict=transform_dict, 
        ssl=True
    )

    train_l_ds = ds_dict['train']['labeled']
    train_u_ds = ds_dict['train']['unlabeled']
    val_ds = ds_dict['val']

    (train_l_loader, train_u_loader), val_loader, test_loader = get_dataloaders(
        train_l_ds=ds_dict['train']['labeled'],
        val_ds=ds_dict['val'],
        train_u_ds=ds_dict['train']['unlabeled'],
        batch_size=args.batch_size
    )

    # args, model, train_l_loader: DataLoader, train_u_loader: DataLoader, val_loader: Dataloader, test_loader: DataLoader=None, filter_bias_and_bn=True
    model = deeplabv3_plus(num_classes=args.num_classes).to(args.device)

    train(args, model, train_l_loader, train_u_loader, val_loader)

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

    print(args)

    main(args)