import torch
import argparse
import yaml
from src.train import train, predict
from src.datasets.fixmatch_datasets import get_datasets
from src.datasets.dataloaders import get_dataloaders
from src.models.create_models import deeplabv3_plus
from src.utils.transforms import train_tfms, strong_tfms, weak_tfms, null_tfms


def main(args):

    transform_dict = {
        'train': train_tfms,
        'strong': strong_tfms,
        'weak': weak_tfms,
        'val': null_tfms,
        'test': null_tfms
    }

    ds_dict = get_datasets(
        train_l_dir=args.train_l_dir, 
        train_u_dir=args.train_u_dir, 
        val_dir=args.val_dir, 
        test_dir=args.test_dir, 
        transform_dict=transform_dict, 
        ssl=args.ssl
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

    model = deeplabv3_plus(args).to(args.device)

    trained_model = train(args, model, train_l_loader, train_u_loader, val_loader)

    predict(args, trained_model, val_loader)

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