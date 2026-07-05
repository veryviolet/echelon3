from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1x1(in_channels, out_channels, stride=1, bias=False):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                    padding=0, bias=bias)


class Activation(nn.Module):
    def __init__(self, act_type, **kwargs):
        super(Activation, self).__init__()
        activation_hub = {'relu': nn.ReLU, 'relu6': nn.ReLU6,
                          'leakyrelu': nn.LeakyReLU, 'prelu': nn.PReLU,
                          'celu': nn.CELU, 'elu': nn.ELU,
                          'hardswish': nn.Hardswish, 'hardtanh': nn.Hardtanh,
                          'gelu': nn.GELU, 'glu': nn.GLU,
                          'selu': nn.SELU, 'silu': nn.SiLU,
                          'sigmoid': nn.Sigmoid, 'softmax': nn.Softmax,
                          'tanh': nn.Tanh, 'none': nn.Identity,
                          }

        act_type = act_type.lower()
        if act_type not in activation_hub.keys():
            raise NotImplementedError(f'Unsupport activation type: {act_type}')

        self.activation = activation_hub[act_type](**kwargs)

    def forward(self, x):
        return self.activation(x)


# Regular convolution -> batchnorm -> activation
class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 bias=False, act_type='relu', **kwargs):
        if isinstance(kernel_size, list) or isinstance(kernel_size, tuple):
            padding = ((kernel_size[0] - 1) // 2 * dilation, (kernel_size[1] - 1) // 2 * dilation)
        elif isinstance(kernel_size, int):
            padding = (kernel_size - 1) // 2 * dilation

        super(ConvBNAct, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            Activation(act_type, **kwargs)
        )


class LightHead(nn.Sequential):

    def __init__(self,
                 in_channels,
                 hidden_channels,
                 num_classes,
                 act_type,
                 **kwargs):
        super(LightHead, self).__init__(
            ConvBNAct(in_channels, hidden_channels, 3, act_type=act_type),
            ConvBNAct(hidden_channels, hidden_channels//2, 3, act_type=act_type),
            conv1x1(hidden_channels//2, num_classes)
        )

