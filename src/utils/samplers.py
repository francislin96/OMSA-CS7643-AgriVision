# Batching sampler for dataloaders to ensure the same size

import torch
from torch.utils.data import DataLoader, Sampler

class BatchSampler(Sampler):
    def __init__(self, dataset_size, total_batches, batch_size):
        self.dataset_size = dataset_size
        self.total_batches = total_batches
        self.batch_size = batch_size
        
    def __iter__(self):
        if self.dataset_size < self.total_batches * self.batch_size:
            # Use modulo operation to ensure indices wrap around dataset size
            indices = torch.randint(0, self.dataset_size, (self.total_batches * self.batch_size,))
            indices = indices % self.dataset_size  # Ensure wrapping around
            return iter(indices.tolist())
        else:
            # When dataset is larger, ensure sampled indices are within bounds
            return iter(torch.randperm(self.dataset_size)[:self.total_batches * self.batch_size].tolist())

    
    def __len__(self):
        return self.total_batches * self.batch_size