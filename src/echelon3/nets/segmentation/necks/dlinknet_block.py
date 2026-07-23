"""D-LinkNet-style dilated central block, wrapping FPN-like neck.

D-LinkNet idea (Zhou DeepGlobe 2018): parallel dilated 3x3 convs with
dilations (1,2,4,8) are applied to the deepest (most semantic)
feature map BEFORE fusion in the FPN. The effect: an enlarged receptive field along long
linear structures (roads, rivers), without increasing the number of samples or losing
spatial resolution.

Here it is a wrapper around FPNLikeNeck: it takes the same list of features, applies the dilated
block to feats[-1] (deepest), then does the regular FPN merge.

Few parameters: 4 × 3x3 conv (out_channels × out_channels) + 1 projection 1x1.
For 960→64 this is ≈ 4 × 64×64×9 = 147k parameters — a pittance.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class DLinkNetCenterBlock(nn.Module):
    """Parallel dilated 3x3 convs + merge via summation.
    in_channels == out_channels; the spatial size is preserved."""

    def __init__(self, channels: int, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.dilations = tuple(int(d) for d in dilations)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            for d in self.dilations
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for b in self.branches:
            out = out + b(x)
        return out


class DLinkNetFPNLikeNeck(nn.Module):
    """FPN-like neck + D-LinkNet dilated center block over the deepest feature.

    Behavior is compatible with FPNLikeNeck (same in/out); feats[-1] simply
    passes through DLinkNetCenterBlock first, after a 1x1 projection
    into the shared out_channels."""

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        align_corners: bool = False,
        dilations=(1, 2, 4, 8),
        **kwargs,
    ):
        super().__init__()
        self.projs = nn.ModuleList(
            [nn.Conv2d(c, out_channels, kernel_size=1, bias=False) for c in in_channels]
        )
        self.center = DLinkNetCenterBlock(out_channels, dilations=dilations)
        self.bn = nn.BatchNorm2d(out_channels)
        self.align_corners = align_corners

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        projected = [p(f) for p, f in zip(self.projs, feats)]
        # D-LinkNet block on the deepest (last) feature — RF expansion.
        projected[-1] = self.center(projected[-1])
        target_size = projected[0].shape[2:]
        out = projected[0]
        for p in projected[1:]:
            out = out + F.interpolate(
                p, size=target_size, mode="bilinear", align_corners=self.align_corners
            )
        return self.bn(out)
