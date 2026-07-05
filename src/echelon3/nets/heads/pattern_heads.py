import os

import torch
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig
from echelon3.creator import create_universal


class MaxPatternHead(nn.Module):

    def __init__(self, **kwargs):
        super(MaxPatternHead, self).__init__()

    def forward(self, x):
        x, y = torch.max(x, dim=-1)
        return x


class AveragePatternHead(nn.Module):

    def __init__(self, **kwargs):
        super(AveragePatternHead, self).__init__()

    def forward(self, x):
        x = torch.mean(x, dim=-1)
        return x


class LinearPatternHead(nn.Module):

    def __init__(self,
                 patterns_count: int,
                 num_classes: int,
                 **kwargs):
        super(LinearPatternHead, self).__init__()
        self.patterns_count = patterns_count
        self.num_classes = num_classes

        self.classifier_layer = nn.Linear(
            in_features=self.num_features,
            out_features=self.num_classes,
            )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier_layer(x)
        return x
