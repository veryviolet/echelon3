"""Losses for auxiliary heads of MultiBinarySegmenterPlus.

Aux head supervision (по образцу RoadNet Liu TGRS 2019 + Batra CVPR 2019):
  - edge        — BCE на Sobel(mask) >= threshold (binary edge GT)
  - centerline  — BCE/MSE на gaussian-blurred skeleton(mask) (soft centerline)
  - orientation — CE per-pixel на discretized tangent angle bin, валидно только
                  по пикселям centerline (остальные — ignored).

Каждый лосс работает по списку ключей "{head}.{aux_type}" в predictions+labels.
ignore_index (255) маскирует не-валидные пиксели.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAuxEdgeBCE(nn.Module):
    """BCE на edge-головах; ожидает labels["{head}.edge"] ∈ {0, 1, 255}."""

    def __init__(self, head_aux_names: List[str], ignore_index: int = 255,
                 head_weights: Dict[str, float] | None = None, **kwargs):
        super().__init__()
        self.keys = list(head_aux_names)
        self.ignore_index = int(ignore_index)
        self.head_weights = {k: float(head_weights.get(k, 1.0)) if head_weights else 1.0
                             for k in self.keys}

    def forward(self, predictions, labels):
        total = next(iter(predictions.values())).new_zeros(())
        denom = 0.0
        for k in self.keys:
            if k not in predictions or k not in labels:
                continue
            logits = predictions[k]
            tgt = labels[k]
            if logits.dim() == 4 and logits.size(1) == 1:
                logits = logits.squeeze(1)
            valid = tgt != self.ignore_index
            if int(valid.sum().item()) == 0:
                continue
            l = F.binary_cross_entropy_with_logits(
                logits[valid], tgt[valid].float(), reduction="mean",
            )
            w = self.head_weights[k]
            total = total + w * l
            denom += w
        if denom == 0.0:
            return total
        return total / denom


class MultiHeadAuxCenterlineBCE(nn.Module):
    """BCE на centerline-головах. Labels — soft float ∈ [0, 1] или {0,1,255}.
    Если labels float ∈ [0,1] — используем как soft target. Если long — как binary."""

    def __init__(self, head_aux_names: List[str], ignore_index: int = 255,
                 head_weights: Dict[str, float] | None = None, **kwargs):
        super().__init__()
        self.keys = list(head_aux_names)
        self.ignore_index = int(ignore_index)
        self.head_weights = {k: float(head_weights.get(k, 1.0)) if head_weights else 1.0
                             for k in self.keys}

    def forward(self, predictions, labels):
        total = next(iter(predictions.values())).new_zeros(())
        denom = 0.0
        for k in self.keys:
            if k not in predictions or k not in labels:
                continue
            logits = predictions[k]
            tgt = labels[k]
            if logits.dim() == 4 and logits.size(1) == 1:
                logits = logits.squeeze(1)
            if tgt.dtype in (torch.long, torch.int64, torch.int32):
                valid = tgt != self.ignore_index
                if int(valid.sum().item()) == 0:
                    continue
                tgt_f = tgt.float()
                l = F.binary_cross_entropy_with_logits(
                    logits[valid], tgt_f[valid], reduction="mean",
                )
            else:
                # soft target ∈ [0, 1]
                l = F.binary_cross_entropy_with_logits(logits, tgt.float(), reduction="mean")
            w = self.head_weights[k]
            total = total + w * l
            denom += w
        if denom == 0.0:
            return total
        return total / denom


class MultiHeadAuxOrientationCE(nn.Module):
    """CE на orientation-головах. Labels — long ∈ [0, num_bins-1] ∪ {ignore_index}."""

    def __init__(self, head_aux_names: List[str], ignore_index: int = 255,
                 head_weights: Dict[str, float] | None = None, **kwargs):
        super().__init__()
        self.keys = list(head_aux_names)
        self.ignore_index = int(ignore_index)
        self.head_weights = {k: float(head_weights.get(k, 1.0)) if head_weights else 1.0
                             for k in self.keys}

    def forward(self, predictions, labels):
        total = next(iter(predictions.values())).new_zeros(())
        denom = 0.0
        for k in self.keys:
            if k not in predictions or k not in labels:
                continue
            logits = predictions[k]   # (B, N, H, W)
            tgt = labels[k].long()    # (B, H, W)
            l = F.cross_entropy(logits, tgt, ignore_index=self.ignore_index, reduction="mean")
            w = self.head_weights[k]
            total = total + w * l
            denom += w
        if denom == 0.0:
            return total
        return total / denom
