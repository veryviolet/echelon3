"""D-LinkNet-style dilated central block, wrapping FPN-like neck.

Идея D-LinkNet (Zhou DeepGlobe 2018): параллельные dilated-3x3-convs с
дилатациями (1,2,4,8) применяются к самой глубокой (наиболее семантической)
фичемапе ДО fusion в FPN. Эффект: расширенное receptive field вдоль длинных
линейных структур (дороги, реки), без увеличения числа сэмплов или потери
пространственного разрешения.

Здесь — обёртка над FPNLikeNeck: принимает тот же список фич, применяет dilated
блок к feats[-1] (deepest), затем штатный FPN-merge.

Параметров мало: 4 × 3x3 conv (out_channels × out_channels) + 1 проекционный 1x1.
Для 960→64 это ≈ 4 × 64×64×9 = 147k параметров — копейка.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class DLinkNetCenterBlock(nn.Module):
    """Параллельные dilated-3x3-convs + объединение через сумму.
    in_channels == out_channels; пространственный размер сохраняется."""

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
    """FPN-like neck + D-LinkNet dilated центр-блок над глубочайшей фичей.

    Поведение совместимо с FPNLikeNeck (тот же in/out), просто feats[-1]
    предварительно проходит через DLinkNetCenterBlock после проекции 1x1
    в общий out_channels."""

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
        # D-LinkNet блок на самой глубокой (последней) фиче — расширение RF.
        projected[-1] = self.center(projected[-1])
        target_size = projected[0].shape[2:]
        out = projected[0]
        for p in projected[1:]:
            out = out + F.interpolate(
                p, size=target_size, mode="bilinear", align_corners=self.align_corners
            )
        return self.bn(out)
