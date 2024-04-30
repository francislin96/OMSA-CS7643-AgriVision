import torch
from tqdm import tqdm
from src.datasets.datasets import *
import argparse
from src.utils.funcs import *
import segmentation_models_pytorch as smp
import os
from collections import OrderedDict
from torch.utils.data import DataLoader

def get_pseudo_labels(net, unlabeled_loader,  threshold=0.5):
    net.eval()
    net.cuda()
    pseudo_labels_all = []

    with torch.no_grad():
        for ui, (inputs, labels) in enumerate(tqdm(unlabeled_loader)):
            inputs = inputs.cuda()
            outputs = net(inputs)
            p_labels = torch.softmax(outputs, dim=1)
            p_labels[p_labels <= threshold] = 0
            label = p_labels.data.max(1)[1].squeeze(1).squeeze(0).cpu()
            pseudo_labels_all.append(label)
    return pseudo_labels_all


def main(args):

    
    TRAIN_ROOT = os.path.join(args.dataset_root,'train')
    VAL_ROOT = os.path.join(args.dataset_root,'val')
    TEST_ROOT = os.path.join(args.dataset_root,'test')
    UNLABELED_ROOT = "./data/images_2024/train"
    
    train_args = agriculture_configs(args, net_name='DeepLabV3plus', data='Agriculture', bands_list=['NIR', 'RGB'], kf=0, k_folder=0, note='Generate_Pseudo_Labels')
     
    _,_,_, unlabeled_set,_ = train_args.get_dataset(args)
    unlabeled_loader = DataLoader(dataset=unlabeled_set, batch_size=2, num_workers=2)

    cpt_path = os.path.join(args.ckpt_path,'DeepLabV3plus/epoch_19_loss_0.45038_acc_0.82621_acc-cls_0.74087_mean-iu_0.60962_fwavacc_0.70690_f1_0.74885_lr_0.0000671535.pth')

    model = smp.DeepLabV3Plus(encoder_name=args.encoder,encoder_weights=args.encoder_weights,encoder_depth=args.encoder_depth, classes=args.num_classes,in_channels=args.input_channels)
    checkpoint = torch.load(cpt_path)
    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        name = k[7:]
        new_state_dict[name]=v
    model.load_state_dict(new_state_dict)
    pseudo_labels = get_pseudo_labels(model, unlabeled_loader, 0.6)
    
    flat_pseudo_labels = torch.stack([arr for element in pseudo_labels for arr in element],dim=0)
    file_path = "pseudo_labels_tensor.pt"
    torch.save(flat_pseudo_labels, file_path)

    print(f"The pseudo labels were saved to a torch file {file_path}")

if __name__ == '__main__':
           
    parser = argparse.ArgumentParser(description='Train a segmentation model using SSL')
    parser.add_argument('-config', type=str, help='Path to the run yaml configuration file', required=True)

    args, unknown = parser.parse_known_args()

    config = load_yaml_config(str(args.config))

    args = set_args_attr(config, args)
    main(args)
