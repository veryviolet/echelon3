import torch
from abc import abstractmethod
from typing import Dict
from torch.utils.data import DataLoader


class Evaluator:

    net: Dict[str, torch.nn.Module] = None
    dataloader: DataLoader = None
    metric = None
    device = None
    preprocess: Dict[str, torch.nn.Module] = None
    postprocess: Dict[str, torch.nn.Module] = None

    def __init__(self, net, dataloader: DataLoader, metric, preprocess, postprocess, **kwargs):
        self.net = net
        self.dataloader = dataloader
        self.metric = metric
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.preprocess = preprocess
        self.postprocess = postprocess

    def evaluate(self):
        return self.evaluate_one(self.dataloader, 'test')

    @abstractmethod
    def evaluate_one(self, dataloader, mode: str):
        pass