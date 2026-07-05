from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self,
                 channels,
                 num_classes,
                 **kwargs):
        super(SimpleHead, self).__init__()

        self.channels = channels
        self.num_classes = num_classes

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)

    def forward(self, features: torch.Tensor):
        x = self.conv_seg(features)
        return x


class SimpleAggregatingHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self,
                 in_channels,
                 channels,
                 num_classes,
                 in_index,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=dict(type='ReLU'),
                 ignore_index=255,
                 sampler=None,
                 align_corners=False,
                 is_dw=False,
                 **kwargs):
        super(SimpleAggregatingHead, self).__init__()

        self.channels = channels
        self.in_channels = list(in_channels)
        self.in_index = list(in_index)
        self.num_classes = num_classes
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.ignore_index = ignore_index
        self.align_corners = align_corners

        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        self.dropout = None
        self.fp16_enabled = False

        embedding_dim = self.channels

        self.linear_fuse = nn.Conv2d(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            kernel_size=1,
            stride=1,
            groups=embedding_dim if is_dw else 1
#            norm_cfg=dict(type='BN', requires_grad=True),
#            act_cfg=self.act_cfg
        )
    
    def agg_res(self, preds: List[torch.Tensor]):
        outs = preds[0]
        for pred in preds[1:]:
            pred = F.interpolate(pred, outs.size()[2:], mode='bilinear', align_corners=False)
            outs += pred
        return outs

    def cls_seg(self, feat):
        """Classify each pixel."""
        output = self.conv_seg(feat)
        return output

    def forward(self, inputs: List[torch.Tensor]):
        xx = [inputs[i] for i in self.in_index]
        x = self.agg_res(xx)
        _c = self.linear_fuse(x)
        x = self.cls_seg(_c)
        return x


