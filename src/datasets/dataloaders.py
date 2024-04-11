# dataloaders.py

import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from src.utils.transforms import collate_fn, unlab_collate_fn
from src.utils.samplers import BatchSampler


def get_dataloaders(train_l_ds: Dataset, val_ds: Dataset, test_ds: Dataset=None, train_u_ds: Dataset=None, batch_size: int=16, num_workers: int=2, collate_fn: tuple=(collate_fn, unlab_collate_fn)):

    # calculate ds lengths and total batches
    lds_size = len(train_l_ds)
    uds_size = len(train_u_ds)

    # Calculate the number of batches required for one epoch with full batches
    total_batches = max(lds_size, uds_size) // batch_size


    train_l_loader = DataLoader(
        train_l_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True, 
        pin_memory=True,
        shuffle=False,
        collate_fn=collate_fn[0],
        sampler=BatchSampler(lds_size, total_batches, batch_size)
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=True, 
        pin_memory=True,
        shuffle=False,
        collate_fn=collate_fn[0]
    )

    if test_ds:
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True, 
            pin_memory=True,
            shuffle=False,
            collate_fn=collate_fn[0]
        )
    else:
        test_loader=None
    
    if train_u_ds:
        train_u_loader = DataLoader(
            train_u_ds, 
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=True,
            shuffle=False,
            collate_fn=collate_fn[1],
            sampler=BatchSampler(lds_size, total_batches, batch_size)
        )
    else:
        train_u_loader = None


    return (train_l_loader, train_u_loader), val_loader, test_loader