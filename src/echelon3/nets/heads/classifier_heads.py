import os

import torch
import torch.nn as nn
from omegaconf import OmegaConf, DictConfig
from echelon3.creator import create_universal

ACTIVATIONS_MODULE_PREFIX = 'echelon3.nets.activations.'


class DenseClassifierHead(nn.Module):

    num_classes: int = None
    num_features: int = None
    dropout: float = None

    def __init__(self,
                 num_classes: int,
                 num_features: int,
                 dropout: float,
                 activation: DictConfig, **kwargs):
        super(DenseClassifierHead, self).__init__()
        self.num_classes = num_classes
        self.num_features = num_features
        self.dropout = dropout

        self.classifier_layer = nn.Linear(
            in_features=self.num_features,
            out_features=self.num_classes,
            )

        self.dropout_layer = nn.Dropout(p=self.dropout)

        self.activation = create_universal(config=activation)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.dropout_layer(x)
        x = self.classifier_layer(x)
        return x


class AgnosticClassifierHead(nn.Module):

    def __init__(self, ord, sigma, quantization_aware=False, **kwargs):
        super(AgnosticClassifierHead, self).__init__()
        self.ord = ord
        self.sigma = sigma
        self.quantization_aware = quantization_aware

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.sqrt(torch.sum(x*x, dim=-1))
#        x = torch.linalg.norm(x, ord=self.ord, dim=-1)
        x = 1.0 - torch.exp(-x*x/self.sigma)
        return x


class Conv1x1ClassifierHead(nn.Module):

    num_channels: int = None
    num_classes: int = None
    kernel_size: tuple = None
    dropout: float = None

    def __init__(self,
                 num_channels: int,
                 num_classes: int,
                 dropout: float,
                 activation: DictConfig,
                 kernel_size: tuple = (1, 1)):
        super(Conv1x1ClassifierHead, self).__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.kernel_size = tuple(kernel_size)

        self.dropout = dropout

        if self.dropout > 0.0:
            self.head_dropout = nn.Dropout2d(p=self.dropout)

        self.head_conv = nn.Conv2d(
            kernel_size=self.kernel_size,
            in_channels=self.num_channels,
            out_channels=self.num_classes)

        self.head_activation = create_universal(config=activation, module_prefix=ACTIVATIONS_MODULE_PREFIX)

    def forward(self, x):
        x = self.head_conv(x)
        if self.dropout > 0.0:
            x = self.head_dropout(x)
        x = self.head_activation(x)
        x = x.view(x.size(0), x.size(1), -1)
        x = torch.mean(x, dim=-1)
        return x



class CrossVitHead(nn.Module):

    num_classes: int = None
    num_features_m: int = None
    num_features_l: int = None
    dropout: float = None


    def __init__(self,
                 num_classes: int,
                 num_features_m: int,
                 num_features_l: int,
                 activation: DictConfig, **kwargs):
        super(CrossVitHead, self).__init__()
        self.num_classes = num_classes
        self.num_features_m = num_features_m
        self.num_features_l = num_features_l

        self.classifier_layer_m = nn.Linear(
            in_features=self.num_features_m,
            out_features=self.num_classes,
            )

        self.classifier_layer_l = nn.Linear(
            in_features=self.num_features_l,
            out_features=self.num_classes,
            )

        self.activation = create_universal(config=activation)

    def forward(self, x):
        m_x = x[0].view(x[0].size(0), -1)
        l_x = x[1].view(x[1].size(0), -1)
        m_scores = torch.unsqueeze(self.classifier_layer_m(m_x), dim=1)
        l_scores = torch.unsqueeze(self.classifier_layer_l(l_x), dim=1)
        res = torch.mean(torch.concat([m_scores, l_scores], dim=1), dim=1)
        return res

class CrossVitAgnosticHead(nn.Module):


    def __init__(self, ord, sigma, quantization_aware=False, **kwargs):
        super(CrossVitAgnosticHead, self).__init__()
        self.ord = ord
        self.sigma = sigma
        self.quantization_aware = quantization_aware


    def forward(self, x):
        x = [f.view(f.size(0), -1) for f in x]
        x = torch.concat(x, dim=-1)
        x = torch.sqrt(torch.sum(x*x, dim=-1))
#        x = torch.linalg.norm(x, ord=self.ord, dim=-1)
        x = 1.0 - torch.exp(-x*x/self.sigma)
        return x

