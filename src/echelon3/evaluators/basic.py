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

    def __init__(self, net, dataloader: DataLoader, metric, preprocess, postprocess,
                 device=None, **kwargs):
        self.net = net
        self.dataloader = dataloader
        self.metric = metric
        if device is None:
            # Direct evaluator users follow their model, never a global cuda:0 default.
            from itertools import chain
            tensor = next(chain(net.parameters(), net.buffers()), None)
            device = tensor.device if tensor is not None else 'cpu'
        self.device = torch.device(device)
        self.preprocess = preprocess
        self.postprocess = postprocess

    def evaluate(self):
        return self.evaluate_one(self.dataloader, 'test')

    @abstractmethod
    def evaluate_one(self, dataloader, mode: str):
        pass
