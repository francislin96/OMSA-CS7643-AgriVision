import os
import argparse
import logging
from tqdm import tqdm
from typing import Callable, Tuple
from PIL import Image
from functools import partial
import matplotlib
matplotlib.use('Agg') # Non GUI backend for matplotlib
import matplotlib.pyplot as plt
import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.utils.eval import *
from src.utils.training import add_weight_decay
from src.models.optimizers import get_exp_scheduler, get_SGD, get_adam
from src.metrics import Metrics
from src.utils.training import reweight_loss
from src.loss.criterions import DiceLoss, TverskyLoss, FocalTverskyLoss
from data.dataset_maps import class_mapping
from src.utils.funcs import *
import torchvision.utils as vutils



def train(
        args, 
        model, 
        train_l_loader: DataLoader, 
        val_loader: DataLoader, 
        criterion: torch.nn.Module,
        filter_bias_and_bn=True
    ):

    logger = logging.getLogger()

    # Set up checkpoint_dir
    chkpt_root_dir = args.checkpoint_dir
    chkpt_path = os.path.join(chkpt_root_dir, args.model_run_name)
    os.makedirs(chkpt_path, exist_ok=True)

    best_val_loss = float('inf')
    
    # Set up main train arguments
    weight_decay = args.weight_decay
    if weight_decay and filter_bias_and_bn:
        parameters = add_weight_decay(model, weight_decay)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    # Get optimizer and scheduler
    if args.optimizer.lower() == 'sgd':
        optimizer = get_SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=weight_decay, nesterov=args.nesterov)
    elif args.optimizer.lower() == 'adam':
        optimizer = get_adam(parameters, lr=args.lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"args.optimizer should be one of ['sgd', 'adam']")

    scheduler = get_exp_scheduler(optimizer, gamma=args.gamma)
    start_epoch = 0

    # Instantiate metrics
    metrics = Metrics(args, class_mapping)
    epoch_meters = AverageMeterSet()
    # Main training loop
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_metrics = train_epoch(
            args,
            model, 
            optimizer, 
            scheduler,
            epoch,
            metrics, 
            train_l_loader,
            criterion
        )
        epoch_meters.update("epoch_train_loss", train_loss, 1)
        print("epoch_train_loss: ", train_loss)
        print(train_metrics)

        val_loss, val_metrics = validate_epoch(
            args,
            model,
            epoch,
            metrics,
            val_loader,
            criterion
        )

        epoch_meters.update("epoch_val_loss", val_loss)
        print("epoch_val_loss: ", val_loss)
        print(val_metrics)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(chkpt_path, f"{args.model_run_name}_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path} at epoch {epoch+1} with loss {val_loss}")

    return model

def train_epoch(
        args, 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        scheduler: torch.optim.lr_scheduler,
        epoch : int,
        metrics: Metrics,
        train_l_loader: DataLoader, 
        criterion: torch.nn.Module
):
    meters = AverageMeterSet()
    metrics.reset()
    model.train()

    p_bar = tqdm(range(len(train_l_loader)))

    for batch_idx, batch in enumerate(train_l_loader):
        # Zero the gradient
        model.zero_grad()

        # Calculate loss
        loss, metrics = train_step(args, model, batch, meters, metrics, criterion)

        # Backpropagate and step optimizer
        loss.backward()
        optimizer.step()
        
        # remove plotting later
        # epoch_losses.append(loss.cpu().item())
        # if batch_idx % 100 == 0:
        #     plt.plot(epoch_losses)
        #     plt.savefig(f'Losses_{batch_idx}.png')
        #     plt.close()

        p_bar.set_description(
            "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Loss: {loss:.6f}".format(
                epoch=epoch + 1,
                epochs=args.epochs,
                batch=batch_idx + 1,
                iter=len(train_l_loader),
                lr=scheduler.get_last_lr()[0],
                loss=loss.item()
            )
        )
        p_bar.update()

    # Move scheduler step to epoch level
    scheduler.step()

    return meters['total_loss'].avg, metrics
    
def train_step(
        args, 
        model: torch.nn.Module, 
        batch: Tuple,
        meters: AverageMeterSet,
        metrics: Metrics,
        criterion: torch.nn.Module
    ):

    img, labels = batch
    img = img.float().to(args.device)
    labels = labels.to(args.device).long()
    logits = model(img)
    if args.focal_loss:
        pass

    # Compute loss and update meters
    loss = criterion(logits, labels)
    meters.update("total_loss", loss.item(), 1)

    # metrics update
    metrics.update_labeled(logits, labels)

    return loss, metrics

@torch.no_grad()
def pseudo_labels(args, logits_u_weak):

    pseudo_labels = torch.softmax(logits_u_weak, dim=1)
    max_probs, targets_u = torch.max(pseudo_labels, dim=1)
    mask = max_probs.ge(args.tau).float()

    return targets_u, mask

@torch.no_grad()
def validate_epoch(
        args, 
        model: torch.nn.Module, 
        epoch : int,
        metrics: Metrics,
        val_loader: DataLoader,
        criterion: torch.nn.Module
):
    model.eval()
    meters = AverageMeterSet()

    p_bar = tqdm(range(len(val_loader)))
    for batch_idx, val_batch in enumerate(val_loader):
        # Calculate loss
        val_loss, metrics = validate_step(args, model, val_batch, meters, metrics, criterion)

        p_bar.set_description(
            "Val Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. Loss: {loss:.6f}".format(
                epoch=epoch + 1,
                epochs=args.epochs,
                batch=batch_idx + 1,
                iter=len(val_loader),
                loss=val_loss.item()
            )
        )
        p_bar.update()

    return meters["val_loss"].avg, metrics

@torch.no_grad()
def validate_step(
        args, 
        model: torch.nn.Module, 
        val_batch: Tuple,
        meters: AverageMeterSet,
        metrics: Metrics,
        criterion: torch.nn.Module
    ):

    val_img, val_labels = val_batch
    val_img = val_img.float().to(args.device)
    val_labels = val_labels.to(args.device).long()
    logits = model(val_img)

    # Compute loss and update meters
    val_loss = criterion(logits, val_labels)
    meters.update("val_loss", val_loss.item(), 1)

    # metrics update
    metrics.update_validation(logits, val_labels)

    return val_loss, metrics

def validate(train_args, args, net, writer, val_set, val_loader, criterion, optimizer, epoch, new_ep):
    from src.metrics import AverageMeter
    net.eval()
    val_loss = AverageMeter()
    inputs_all, gts_all, predictions_all = [], [], []

    with torch.no_grad():
        for vi, (inputs, gts) in enumerate(val_loader):
          
            inputs, gts = inputs.cuda(), gts.cuda()
            N = inputs.size(0) * inputs.size(2) * inputs.size(3)
            outputs = net(inputs)

            val_loss.update(criterion(outputs, gts).item(), N)
            
            if random.random() > args.save_rate:
                inputs_all.append(None)
            else:
                inputs_all.append(inputs.data.squeeze(0).cpu())

            gts_all.append(gts.data.squeeze(0).cpu().numpy())
            predictions = outputs.data.max(1)[1].squeeze(1).squeeze(0).cpu().numpy()
            predictions_all.append(predictions)

    update_ckpt(args, train_args, net, writer, optimizer, epoch, new_ep, val_loss,
                inputs_all, gts_all, predictions_all)
    #visual_ckpt(epoch, new_ep, inputs_all, gts_all, predictions_all)
    net.train()
    return val_loss, inputs_all, gts_all, predictions_all


def update_ckpt(args, train_args, net, writer, optimizer, epoch, new_ep, val_loss,
                inputs_all, gts_all, predictions_all):
    avg_loss = val_loss.avg

    acc, acc_cls, mean_iu, fwavacc, f1,ui = evaluate(predictions_all, gts_all, args.num_classes)

    writer.add_scalar('val_loss', avg_loss, epoch)
    writer.add_scalar('acc', acc, epoch)
    writer.add_scalar('acc_cls', acc_cls, epoch)
    writer.add_scalar('mean_iu', mean_iu, epoch)
    writer.add_scalar('fwavacc', fwavacc, epoch)
    writer.add_scalar('f1_score', f1, epoch)

    updated = train_args.update_best_record(epoch, avg_loss, acc, acc_cls, mean_iu, fwavacc, f1)

    # save best record and snapshot prameters
    val_visual = []

    snapshot_name = 'epoch_%d_loss_%.5f_acc_%.5f_acc-cls_%.5f_mean-iu_%.5f_fwavacc_%.5f_f1_%.5f_lr_%.10f' % (
        epoch, avg_loss, acc, acc_cls, mean_iu, fwavacc, f1, optimizer.param_groups[0]['lr']
    )

    if updated or (args.best_record['val_loss'] > avg_loss):
        torch.save(net.state_dict(), os.path.join(args.save_path, snapshot_name + '.pth'))
        # train_args.update_best_record(epoch, val_loss.avg, acc, acc_cls, mean_iu, fwavacc, f1)
    #if args.save_pred:
    #    if updated or (new_ep % 5 == 0):
    #        val_visual = visual_ckpt(epoch, new_ep, inputs_all, gts_all, predictions_all)

    if len(val_visual) > 0:
        val_visual = torch.stack(val_visual, 0)
        val_visual = vutils.make_grid(val_visual, nrow=3, padding=5)
        writer.add_image(snapshot_name, val_visual)

