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

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.utils.eval import AverageMeterSet
from src.utils.training import add_weight_decay
from src.models.optimizers import get_exp_scheduler, get_SGD, get_adam
from src.metrics import Metrics
from src.utils.training import reweight_loss
from src.loss.criterions import DiceLoss, TverskyLoss, FocalTverskyLoss

logger = logging.getLogger()

def train(
        args, 
        model, 
        train_l_loader: DataLoader, 
        val_loader: DataLoader, 
        train_u_loader: DataLoader=None, 
        filter_bias_and_bn=True
    ):
    
    weight_decay = args.weight_decay
    if weight_decay and filter_bias_and_bn:
        parameters = add_weight_decay(model, weight_decay)
        weight_decay = 0.
    else:
        parameters = model.parameters()

    if args.optimizer.lower() == 'sgd':
        optimizer = get_SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=weight_decay, nesterov=args.nesterov)
    elif args.optimizer.lower() == 'adam':
        optimizer = get_adam(parameters, lr=args.lr, weight_decay=weight_decay)

    scheduler = get_exp_scheduler(optimizer, gamma=args.gamma)
    start_epoch = 0


    metrics = Metrics(args)
    for epoch in range(start_epoch, args.epochs):
        if train_u_loader:
            train_total_loss, train_l_loss, train_u_loss, metrics = train_ssl_epoch(
                args,
                model,
                optimizer,
                scheduler,
                epoch, 
                metrics,
                train_l_loader,
                train_u_loader
            )

            print("total_loss: ", train_total_loss)
            print("labeled_loss: ", train_l_loss)
            print("unlabeled_loss: ", train_u_loss)
            print(metrics)

        else: 
            train_loss, metrics = train_epoch(
                args,
                model, 
                optimizer, 
                scheduler,
                epoch,
                metrics, 
                train_l_loader
            )
            print("train_loss: ", train_loss)
            print("validation loss: ", val_loss)
            print(metrics)

        val_loss, val_metrics = validate_epoch(model, val_loader, metrics)
        print("val_loss: ", val_loss)
        print(val_metrics)

    return model

def train_epoch(
        args, 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        scheduler: torch.optim.lr_scheduler,
        epoch : int,
        metrics: Metrics,
        train_l_loader: DataLoader, 
):
    meters = AverageMeterSet()
    metrics.reset()

    # model.zero_grad()
    model.train()

    epoch_losses = [] # remove later
    p_bar = tqdm(range(len(train_l_loader)))

    for batch_idx, batch in enumerate(train_l_loader):
        # Zero the gradient
        model.zero_grad()
        # optimizer.zero_grad()
        # Calculate loss
        loss, metrics = train_step(args, model, batch, meters, metrics, ssl=False)
        # Backpropagate and step optimizer
        loss.backward()
        optimizer.step()
        # scheduler.step()
        
        # remove plotting later
        epoch_losses.append(loss.cpu().item())
        if batch_idx % 100 == 0:
            plt.plot(epoch_losses)
            plt.savefig(f'Losses_{batch_idx}.png')
            plt.close()

        p_bar.set_description(
            "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}.".format(
                epoch=epoch + 1,
                epochs=args.epochs,
                batch=batch_idx + 1,
                iter=len(train_l_loader),
                lr=scheduler.get_last_lr()[0],
            )
        )
        p_bar.update()

    # Move scheduler step to epoch level
    scheduler.step()
    if args.ssl:
        return (
            meters["total_loss"].avg,
            meters["labeled_loss"].avg,
            meters["unlabeled_loss"].avg,
        )
    else:
        return meters['total_loss'].avg
    
def train_ssl_epoch(
        args, 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        scheduler: torch.optim.lr_scheduler,
        epoch : int,
        metrics: Metrics,
        train_l_loader: DataLoader, 
        val_loader: DataLoader,
        train_u_loader: DataLoader
    ):

    meters = AverageMeterSet()
    metrics.reset()
    model.zero_grad()
    model.train()

    epoch_losses = [] # remove later
    p_bar = tqdm(range(len(train_l_loader)))
    
    for batch_idx, batch in enumerate(
        zip(train_l_loader, train_u_loader)
    ):
        optimizer.zero_grad()
        loss, metrics = train_step(args, model, batch, meters, metrics, ssl=True)
        loss.backward()
        optimizer.step()
        # scheduler.step()

        # remove plotting later
        epoch_losses.append(loss.cpu().item())
        if batch_idx % 100 == 0:
            plt.plot(epoch_losses)
            plt.savefig(f'Losses_{batch_idx}.png')
            plt.close()

        p_bar.set_description(
            "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}.".format(
                epoch=epoch + 1,
                epochs=args.epochs,
                batch=batch_idx + 1,
                iter=len(train_l_loader),
                lr=scheduler.get_last_lr()[0],
            )
        )
        p_bar.update()

    # Move scheduler step to epoch level
    scheduler.step()

    return (
        meters["total_loss"].avg,
        meters["labeled_loss"].avg,
        meters["unlabeled_loss"].avg,
    )

def train_step(
        args, 
        model: torch.nn.Module, 
        batch: Tuple,
        meters: AverageMeterSet,
        metrics: Metrics,
        ssl: bool=False
    ):

    # unpack batches for ssl
    if ssl:
        lab_batch, unlab_batch = batch
        l_img, labels = lab_batch
        weak_img, strong_img = unlab_batch

        # Concatenate inputs and send to device
        inputs = torch.cat((l_img, weak_img, strong_img)).float().to(args.device)
        labels = labels.to(args.device).long()

        # Compute logits for labeled and unlabeled data
        logits = model(inputs)
        logits_x = logits[:len(l_img)]
        logits_u_weak, logits_u_strong = logits[len(l_img):].chunk(2)
        del inputs

        # Compute CE loss for labeled samples
        if args.focal_loss:
            labeled_loss = F.cross_entropy(logits_x, labels, reduction="mean", weight=reweight_loss(labels))
        else:
            labeled_loss = F.cross_entropy(logits_x, labels, reduction="mean")

        # Compute pseudo-labels for unlabeled samples based on model predictions on weakly augmented samples
        targets_u, mask = pseudo_labels(args, logits_u_weak)

        # Calculate CE loss between pseudo labels and strong augmentation logits
        unlabeled_loss = (F.cross_entropy(logits_u_weak, targets_u, reduction="none") * mask).mean() * 1/args.mu

        loss = labeled_loss.mean() + args.lam * unlabeled_loss

        print("Losses: ", loss.item(), labeled_loss.item(), unlabeled_loss.item())

        meters.update("total_loss", loss.item(), 1)
        meters.update("labeled_loss", labeled_loss.mean().item(), logits_x.size()[0])
        meters.update("unlabeled_loss", unlabeled_loss.item(), logits_u_strong.size()[0])

        # metrics
        metrics.update_labeled(logits_x, labels)
        metrics.update_unlabeled(logits_u_weak, targets_u)

    else:
        img, labels = batch
        img = img.float().to(args.device)
        labels = labels.to(args.device).long()
        logits = model(img)
        if args.focal_loss:
            loss = F.cross_entropy(logits, labels, reduction="mean", weight=reweight_loss(labels))
        else:
            dice_loss_criterion = DiceLoss(classes=args.num_classes, ignore_index=0)
            tversky_loss_criterion = TverskyLoss(args, alpha=0.5, beta=0.5)
            focal_tversky = FocalTverskyLoss(args)

            # loss = DiceLoss(logits, labels, reduction="mean")
            # loss = dice_loss_criterion(logits, labels)
            loss = focal_tversky(logits, labels)
        
        print("Losses: ", loss.item())
        
        meters.update("total_loss", loss.item(), 1)

        # metrics
        metrics.update_labeled(logits, labels)

    return loss, metrics

@torch.no_grad()
def pseudo_labels(args, logits_u_weak):

    pseudo_labels = torch.softmax(logits_u_weak, dim=1)
    max_probs, targets_u = torch.max(pseudo_labels, dim=1)
    mask = max_probs.ge(args.tau).float()

    return targets_u, mask

@torch.no_grad()
def validate_epoch(args, model, val_loader, metrics):
    model.eval()
    val_loss = 0
    val_size = 0
    for val_batch in val_loader:
        val_img, val_labels = val_batch
        val_img = val_img.to(args.device)
        val_labels = val_img.to(args.device).long()
        logits = model(val_img)
        metrics.update_validation(logits, val_labels)
        val_loss += F.cross_entropy(logits, val_labels.long(), reduction="mean") * val_img.size(0)
        val_size += val_img.size(0)
    # average loss
    return (val_loss / val_size).item(), metrics