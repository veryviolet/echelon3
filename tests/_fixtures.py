"""Tiny classes for create_* / Trainer tests without external data."""
import torch
from torch.utils.data import Dataset, DataLoader


class TinyPairs(Dataset):
    """A dataset that accepts augment/preprocess (as create_single_dataset expects)."""

    def __init__(self, n: int = 8, augment=None, preprocess=None):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        x = torch.zeros(4)
        return x, x.clone()


def tiny_loader(n: int = 8, batch_size: int = 4):
    return DataLoader(TinyPairs(n), batch_size=batch_size)
