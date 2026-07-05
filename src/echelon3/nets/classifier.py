import os
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig
from echelon3.creator import create_backbone, create_universal
import torch

HEADS_MODULE_PREFIX = 'echelon3.nets.heads.'


class ClassifierNet(nn.Module):

    num_classes: int = None
    num_features: int = None
    dropout: float = None

    def __init__(self,
                 backbone: DictConfig,
                 head: DictConfig,
                 quantization_aware=False,
                 **kwargs):
        super(ClassifierNet, self).__init__()

        self.backbone = create_backbone(backbone)
        self.head = create_universal(head)

    def reread_hypers(self, **kwargs):
        pass

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

