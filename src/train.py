import os
import argparse
import logging
from tqdm import tqdm
from typing import Callable, Tuple
from PIL import Image
from functools import partial
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from src.utils.eval import AverageMeterSet
from src.utils.training import add_weight_decay
from src.models.optimizers import get_exp_scheduler, get_SGD
from src.metrics import Metrics
from src.utils.training import reweight_loss


logger = logging.getLogger()

def train(
        args, 
        model, 
        train_l_loader: DataLoader, 
        train_u_loader: DataLoader, 
        val_loader: DataLoader, 
        test_loader: DataLoader=None, 
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

    scheduler = get_exp_scheduler(optimizer, gamma=args.gamma)
    start_epoch = 0

    metrics = Metrics(args)
    for epoch in range(start_epoch, args.epochs):
        train_total_loss, train_l_loss, train_u_loss, val_loss, metrics = train_epoch(
            args,
            model,
            optimizer,
            scheduler,
            train_l_loader,
            train_u_loader,
            val_loader,
            epoch,
            metrics
        )
        print("total_loss: ", train_total_loss)
        print("labeled_loss: ", train_l_loss)
        print("unlabeled_loss: ", train_u_loss)
        print("validation loss: ", val_loss)
        print(metrics)

    return model


def train_epoch(
        args, 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer, 
        scheduler: torch.optim.lr_scheduler,
        train_l_loader, 
        train_u_loader,
        val_loader,
        epoch,
        metrics
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
        loss, metrics = train_step(args, model, batch, meters, metrics)

        # remove plotting later
        epoch_losses.append(loss.cpu().item())
        if batch_idx % 100 == 0:
            plt.plot(epoch_losses)
            plt.savefig(f'Losses_{batch_idx}.png')
            plt.close()


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

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

    val_loss, metrics = validate_epoch(model, val_loader, metrics)
    print(metrics)

    return (
        meters["total_loss"].avg,
        meters["labeled_loss"].avg,
        meters["unlabeled_loss"].avg,
        val_loss,
        metrics
    )


def train_step(
        args, 
        model: torch.nn.Module, 
        batch: Tuple,
        # val_loader: DataLoader,
        meters: AverageMeterSet,
        metrics: Metrics
    ):

    # unpack batches
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

    # for val_batch in val_loader:
    #     val_img, val_labels = val_batch
    #     metrics.update_validation(model(val_img), val_labels)
    # print(metrics)

    return loss, metrics

@torch.no_grad()
def pseudo_labels(args, logits_u_weak):

    pseudo_labels = torch.softmax(logits_u_weak, dim=1)
    max_probs, targets_u = torch.max(pseudo_labels, dim=1)
    mask = max_probs.ge(args.tau).float()

    return targets_u, mask

@torch.no_grad()
def validate_epoch(model, val_loader, metrics):
    model.eval()
    val_loss = 0
    val_size = 0
    for val_batch in val_loader:
        val_img, val_labels = val_batch
        logits = model(val_img)
        metrics.update_validation(logits, val_labels)
        val_loss += F.cross_entropy(logits, val_labels.long(), reduction="mean") * val_img.size(0)
        val_size += val_img.size(0)
    # average loss
    return (val_loss / val_size).item(), metrics

@torch.no_grad()
def predict(args, model, data_loader, visualize=True):
    model.eval()
    name = 0
    for batch in data_loader:
        img, labels = batch
        img = img.to(args.device)
        labels = labels.to(args.device).long()
        logits = model(img)
        predictions = torch.argmax(logits, dim=1)
        
        if visualize:
            from src.utils.visualize import display_image
            for i in range(img.size(0)):
                display_image(
                    rgb_image=img[i, :3, :, :].permute(1 , 2, 0).cpu().numpy(),
                    nir_image=img[i, 3, :, :].unsqueeze(2).cpu().numpy(),
                    true_labels=labels[i, :, :].unsqueeze(2).cpu().numpy(),
                    pred_labels=predictions[i, :, :].cpu().numpy(),
                    filename=str(name)+".jpg"
                )
                name += 1