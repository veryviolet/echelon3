import torch
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler
from torchvision import datasets
from torchvision.transforms import ToTensor
from omegaconf import OmegaConf, DictConfig, ListConfig
import cv2
import numpy as np
from random import shuffle as shuffle


def collate_fn(batch):
    return tuple(zip(*batch))


from echelon3.data.basic import PerClassFilesDataset


class VariableDataLoader(DataLoader):
    def __init__(self, **kwargs):
        super(VariableDataLoader, self).__init__(collate_fn=collate_fn, **kwargs)

