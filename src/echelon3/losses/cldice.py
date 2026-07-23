"""clDice (centerline Dice) — Shit et al. CVPR 2021, https://arxiv.org/abs/2003.07311.

A topology-aware loss for thin tubular/linear structures (roads, rivers,
shelterbelts). It directly penalizes breaks in the centerline.

Idea: a soft skeleton over k iterations (min-pool then max-pool on the soft prediction).
Then clDice = 2 * (T_prec * T_sens) / (T_prec + T_sens) where
  T_prec  = sum(skel_pred * gt)     / sum(skel_pred)
  T_sens  = sum(skel_gt   * pred)   / sum(skel_gt)

Here it is the multi-head variant (like the other multibinary losses in echelon3).
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _soft_erode(x: torch.Tensor) -> torch.Tensor:
    # for 2D: min-pool via -max-pool on the negative
    return -F.max_pool2d(-x, kernel_size=(3, 3), stride=1, padding=1)


def _soft_dilate(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x, kernel_size=(3, 3), stride=1, padding=1)


def _soft_open(x: torch.Tensor) -> torch.Tensor:
    return _soft_dilate(_soft_erode(x))


def soft_skeleton(prob: torch.Tensor, iters: int = 3) -> torch.Tensor:
    """Differentiable skeleton: iteratively subtract the "opening" from the erosion.
    The more iters, the thinner the skeleton (choose it for the max line radius)."""
    img = prob
    img1 = _soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iters):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


class MultiHeadSoftCLDiceWithIgnore(nn.Module):
    """Soft clDice loss for a multi-binary head with ignore_index masking.

    The `smooth` argument stabilizes the division, usually 1.0.
    """

    def __init__(
        self,
        head_names: List[str],
        ignore_index: int = 255,
        head_weights: Dict[str, float] | None = None,
        iters: int = 3,
        smooth: float = 1.0,
        downsample: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.head_names = list(head_names)
        self.ignore_index = int(ignore_index)
        self.head_weights = {
            h: float(head_weights.get(h, 1.0)) if head_weights else 1.0
            for h in self.head_names
        }
        self.iters = int(iters)
        self.smooth = float(smooth)
        # Downsample: inside clDice we work at (H/k, W/k) to save memory.
        # The skeleton of thin lines is robust to ↓2 (and often ↓4) — topology is not lost,
        # and there is no memory peak of soft-erode/dilate × iters × heads at full resolution.
        self.downsample = int(downsample)

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        total = predictions[self.head_names[0]].new_zeros(())
        denom = 0.0
        for head in self.head_names:
            if head not in predictions or head not in labels:
                continue
            logits = predictions[head]
            tgt = labels[head]
            if logits.dim() == 4 and logits.size(1) == 1:
                logits = logits.squeeze(1)
            valid = (tgt != self.ignore_index).float()
            if valid.sum() == 0:
                continue
            prob = torch.sigmoid(logits) * valid
            gt = (tgt.float() * valid).clamp(0.0, 1.0)
            if gt.sum() == 0:
                continue
            prob = prob.unsqueeze(1)
            gt = gt.unsqueeze(1)
            if self.downsample > 1:
                prob = F.avg_pool2d(prob, self.downsample)
                gt = F.max_pool2d(gt, self.downsample)
            skel_pred = soft_skeleton(prob, iters=self.iters)
            skel_gt = soft_skeleton(gt, iters=self.iters)
            t_prec = (torch.sum(skel_pred * gt) + self.smooth) / (torch.sum(skel_pred) + self.smooth)
            t_sens = (torch.sum(skel_gt * prob) + self.smooth) / (torch.sum(skel_gt) + self.smooth)
            cl_dice = 1.0 - 2.0 * (t_prec * t_sens) / (t_prec + t_sens + 1e-6)
            w = self.head_weights[head]
            total = total + w * cl_dice
            denom += w
        if denom == 0.0:
            return total
        return total / denom
