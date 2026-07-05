import torch.nn as nn
import torch.nn.functional as F
import torch

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

class DASPPneck(nn.Module):

    def __init__(self, in_channels, out_channels, act_type):
        super(DASPPneck, self).__init__()
        hid_channels = in_channels // 5
        last_channels = in_channels - hid_channels * 4
        self.stage1 = ConvBNAct(in_channels, hid_channels, 1, act_type=act_type)
        self.stage2 = ConvBNAct(in_channels, hid_channels, 3, dilation=3, act_type=act_type)
        self.stage3 = ConvBNAct(in_channels, hid_channels, 3, dilation=6, act_type=act_type)
        self.stage4 = ConvBNAct(in_channels, hid_channels, 3, dilation=9, act_type=act_type)
        self.stage5 = nn.Sequential(
                            nn.AdaptiveAvgPool2d(1),
                            conv1x1(in_channels, last_channels)
                        )
        self.conv = ConvBNAct(2*in_channels, out_channels, 1, act_type=act_type)

    def forward(self, x):
        _, x_res, _, x = x
        size_res = x_res.size()[2:]

        size = x.size()[2:]

        x1 = self.stage1(x)
        x2 = self.stage2(x)
        x3 = self.stage3(x)
        x4 = self.stage4(x)
        x5 = self.stage5(x)
        x5 = F.interpolate(x5, size, mode='bilinear', align_corners=True)

        x = self.conv(torch.cat([x, x1, x2, x3, x4, x5], dim=1))

        x = F.interpolate(x, size_res, mode='bilinear', align_corners=True)

        x = torch.cat([x, x_res], dim=1)

        return x


class DASPPneck2(nn.Module):

    def __init__(self, in_channels, out_channels, act_type):
        super(DASPPneck2, self).__init__()
        hid_channels = in_channels // 3
        last_channels = in_channels - hid_channels * 2
        self.stage1 = ConvBNAct(in_channels, hid_channels, 1, act_type=act_type)
        self.stage2 = ConvBNAct(in_channels, hid_channels, 3, dilation=3, act_type=act_type)
        self.stage5 = nn.Sequential(
                            nn.AdaptiveAvgPool2d(1),
                            conv1x1(in_channels, last_channels)
                        )
        self.conv = ConvBNAct(2*in_channels, out_channels, 1, act_type=act_type)

    def forward(self, x):
        x_res, x = x
        size_res = x_res.size()[2:]

        size = x.size()[2:]

        x1 = self.stage1(x)
        x2 = self.stage2(x)
        x5 = self.stage5(x)
        x5 = F.interpolate(x5, size, mode='bilinear', align_corners=True)

        x = self.conv(torch.cat([x, x1, x2, x5], dim=1))

        x = F.interpolate(x, size_res, mode='bilinear', align_corners=True)

        x = torch.cat([x, x_res], dim=1)

        return x


