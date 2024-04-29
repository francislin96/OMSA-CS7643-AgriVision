import os
import time
import sys
import argparse
from data.dataset_maps import palette
from src.loss.criterions import *
from src.models.create_models import deeplabv3_plus
from data.dataset_maps import labels_folder
from src.utils.funcs import *
from src.metrics import *
from src.datasets.datasets import AgricultureDataset, agriculture_configs
from tensorboardX import SummaryWriter
import torch
from torch.utils.data.distributed import DistributedSampler
from src.models.optimizers import *
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from src.train import *


def main(args):
    if args.n_gpus > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        torch.distributed.init_process_group(backend='nccl', rank=0, world_size=1, init_method='env://')
    
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
    prepare_gt(TRAIN_ROOT)
    prepare_gt(VAL_ROOT)
    prepare_gt(TEST_ROOT)
    
    train_set, val_set,_, _,_ = train_args.get_dataset(args)
    
    train_batch_size = args.n_gpus * args.train_batch_per_gpu
    val_batch_size = args.n_gpus * args.val_batch_per_gpu
    args.lr = args.lr / np.sqrt(3)
    writer = SummaryWriter(os.path.join(args.save_path, 'tblog'))
    
    random_seed(args.seeds)
    net, start_epoch = train_args.resume_train(model)
    if args.n_gpus > 1:
        net = torch.nn.DataParallel(net)
    net.cuda()
    net.train()
    if args.n_gpus > 1:
        sampler = DistributedSampler(train_set)
        train_loader = DataLoader(train_set, batch_size=train_batch_size, num_workers=4, sampler=sampler)
    else:
        train_loader = DataLoader(train_set, batch_size=train_batch_size, num_workers=4, shuffle=True)
    
    val_loader = DataLoader(dataset=val_set, batch_size=val_batch_size, num_workers=4)
    params = init_params_lr(net, args)
    base_optimizer = SGD(params, momentum=args.momentum, nesterov=True)
    optimizer = Lookahead(base_optimizer, k=6)
    losses = []
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 60, 1.18e-6)
    new_ep = 0
    for _ in range(args.epochs):
        starttime = time.time()
        train_main_loss = AverageMeter()
        cls_trian_loss = AverageMeter()
        start_lr = args.lr
        args.lr = optimizer.param_groups[0]['lr']
        num_iter = len(train_loader)
        curr_iter = ((start_epoch + new_ep) - 1) * num_iter
        print('---curr_iter: {}, num_iter per epoch: {}---'.format(curr_iter, num_iter))
        
        for i, (inputs, labels) in enumerate(train_loader):
            sys.stdout.flush()

            inputs, labels = inputs.cuda(), labels.cuda(),
            N = inputs.size(0) * inputs.size(2) * inputs.size(3)
            optimizer.zero_grad()
            outputs = net(inputs)
            main_loss = criterion(outputs, labels)
            if args.n_gpus > 1:
                loss = main_loss.mean()
            else: 
                loss = main_loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step(epoch=(start_epoch + new_ep))
            train_main_loss.update(loss.item(), N)
            curr_iter += 1
            losses.append(train_main_loss.avg)
            writer.add_scalar('main_loss', train_main_loss.avg, curr_iter)
            writer.add_scalar('cls_loss', cls_trian_loss.avg, curr_iter)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], curr_iter)

            if (i + 1) % args.print_freq == 0:
                newtime = time.time()

                print('[epoch %d], [iter %d / %d], [loss %.5f, cls %.5f], [lr %.10f], [time %.3f]' %
                      (start_epoch + new_ep, i + 1, num_iter, train_main_loss.avg, 
                       cls_trian_loss.avg,
                       optimizer.param_groups[0]['lr'], newtime - starttime))

                starttime = newtime
        # plt.plot(losses)
        # plt.savefig(f'Losses_{i+1}.png')
        # plt.close()
        print("Starting validation")
        val_start_time = time.time()
        validate(train_args, args, net, writer, val_set, val_loader, criterion, optimizer, start_epoch + new_ep, new_ep)
        print(f"Finished validation, took {time.time() - val_start_time}")
        new_ep += 1        


    
if __name__ == '__main__':
           
    parser = argparse.ArgumentParser(description='Train a segmentation model using SSL')
    parser.add_argument('-config', type=str, help='Path to the run yaml configuration file', required=True)
    parser.add_argument('--dataset_root', type=str, default="./data/images_2021", help="Path to dataset")
    parser.add_argument('--node_size', type=tuple, default=(32,32), help="MSCG model node size")
    parser.add_argument('--best_record', type=dict, default={})
    parser.add_argument('-loader', default=AgricultureDataset, help="Dataset")


    args, unknown = parser.parse_known_args()

    config = load_yaml_config(str(args.config))

    args = set_args_attr(config, args)
    main(args)



