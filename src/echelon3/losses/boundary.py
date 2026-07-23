"""Boundary loss for thin/elongated features (Bokhovkin & Burnaev, 2019).

https://arxiv.org/pdf/1905.07852

A differentiable approximation of L_BD that penalizes the deviation of the
prediction boundaries from GT based on a Sobel-ish gradient (computed via
convolutions with hand-built kernels). Cheap, works well on
thin water and road boundaries.

Implementation: for each head we compute gradient maps of the prediction and GT
(magnitude via a Sobel filter), then compute precision/recall between
them, then F1-ish.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


_SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
_SOBEL_Y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)


def _sobel_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Sobel gradient magnitude, normalized to [0, 1].
    For a binary mask, max sobel ≈ sqrt(2)*4 ≈ 5.66; we divide by 4 and clamp,
    so that precision/recall in the boundary F1 do not exceed 1.0
    (otherwise loss=1-F1 becomes negative and the optimizer "learns in reverse").
    """
    kx = _SOBEL_X.to(x.device, x.dtype)
    ky = _SOBEL_Y.to(x.device, x.dtype)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-8) / 4.0
    return mag.clamp(0.0, 1.0)


def _max_pool(x: torch.Tensor, k: int) -> torch.Tensor:
    return F.max_pool2d(x, kernel_size=k, stride=1, padding=k // 2)


class MultiHeadBoundaryWithIgnore(nn.Module):
    """Boundary F1 loss for a multi-binary head.

    Args:
        theta: radius of the "warm" zone for precision/recall boundary matching
            (we count nearby matches, not only pixel-perfect ones).
        smooth: stabilizer.
    """

    def __init__(
        self,
        head_names: List[str],
        ignore_index: int = 255,
        head_weights: Dict[str, float] | None = None,
        theta: int = 3,
        smooth: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self.head_names = list(head_names)
        self.ignore_index = int(ignore_index)
        self.head_weights = {
            h: float(head_weights.get(h, 1.0)) if head_weights else 1.0
            for h in self.head_names
        }
        self.theta = int(theta)
        self.smooth = float(smooth)

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
            prob = prob.unsqueeze(1)
            gt = gt.unsqueeze(1)
            b_pred = _sobel_magnitude(prob)
            b_gt = _sobel_magnitude(gt)
            # "dilated" boundary for the neighborhood
            b_pred_d = _max_pool(b_pred, k=2 * self.theta + 1)
            b_gt_d = _max_pool(b_gt, k=2 * self.theta + 1)
            precision = (torch.sum(b_pred * b_gt_d) + self.smooth) / (torch.sum(b_pred) + self.smooth)
            recall = (torch.sum(b_gt * b_pred_d) + self.smooth) / (torch.sum(b_gt) + self.smooth)
            f1 = 2 * precision * recall / (precision + recall + 1e-6)
            boundary_loss = 1.0 - f1
            w = self.head_weights[head]
            total = total + w * boundary_loss
            denom += w
        if denom == 0.0:
            return total
        return total / denom
