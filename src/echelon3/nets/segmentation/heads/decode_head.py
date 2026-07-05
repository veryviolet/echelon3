# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn

import omegaconf

from echelon3.nets.ops import resize


class BaseDecodeHead(nn.Module, metaclass=ABCMeta):

    def __init__(self,
                 in_channels,
                 channels,
                 *,
                 num_classes,
                 dropout_ratio=0.1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 in_index,
                 ignore_index=255,
                 sampler=None,
                 align_corners=False):
        super(BaseDecodeHead, self).__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.in_index = list(in_index)
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.in_index = in_index

        self.ignore_index = ignore_index
        self.align_corners = align_corners

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None
        self.fp16_enabled = False

    def extra_repr(self):
        """Extra repr."""
        s = f'input_transform={self.input_transform}, ' \
            f'ignore_index={self.ignore_index}, ' \
            f'align_corners={self.align_corners}'
        return s

    def _init_inputs(self, in_channels, in_index):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.
                'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                None: Only one select feature map is allowed.
        """



    @abstractmethod
    def forward(self, inputs):
        """Placeholder of forward function."""
        pass

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
        output = self.conv_seg(feat)
        return output

