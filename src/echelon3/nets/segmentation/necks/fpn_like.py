"""Tiny FPN-like neck.

Takes a list of multi-scale feature maps (timm `features_only` style) and fuses
them by projecting each to a common `out_channels` and summing on the largest
spatial size (the first feature). Output: single tensor (B, out_channels, H, W).

Cheap enough for RK3588 single-NPU deployment: 4 × 1x1 convs + 3 bilinear
upsamples + 3 adds, ~50–200K params depending on in_channels.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPNLikeNeck(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        align_corners: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.projs = nn.ModuleList(
            [nn.Conv2d(c, out_channels, kernel_size=1, bias=False) for c in in_channels]
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.align_corners = align_corners

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        projected = [p(f) for p, f in zip(self.projs, feats)]
        target_size = projected[0].shape[2:]
        out = projected[0]
        for p in projected[1:]:
            out = out + F.interpolate(
                p, size=target_size, mode="bilinear", align_corners=self.align_corners
            )
        return self.bn(out)
