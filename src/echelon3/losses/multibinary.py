"""Per-head BCE-with-logits loss with ignore_index masking.

Expects:
  predictions: dict[head_name -> (B, 1, H, W) logits]
  labels:      dict[head_name -> (B, H, W) long, values in {0, 1, ignore_index}]

Per head, BCE is computed only on valid pixels (label != ignore_index).
A head with zero valid pixels in the batch contributes 0 to the total.

Supports optional per-head weights.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadBCEWithIgnore(nn.Module):
    def __init__(
        self,
        head_names: List[str],
        ignore_index: int = 255,
        head_weights: Dict[str, float] | None = None,
        pos_weight: Dict[str, float] | None = None,
        **kwargs,
    ):
        super().__init__()
        self.head_names = list(head_names)
        self.ignore_index = int(ignore_index)
        self.head_weights = {h: float(head_weights.get(h, 1.0)) if head_weights else 1.0
                             for h in self.head_names}
        self.pos_weight = pos_weight or {}

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
                logits = logits.squeeze(1)            # (B, H, W)

            valid = tgt != self.ignore_index
            n_valid = int(valid.sum().item())
            if n_valid == 0:
                continue

            pw = None
            if head in self.pos_weight:
                pw = logits.new_tensor([float(self.pos_weight[head])])

            l = F.binary_cross_entropy_with_logits(
                logits[valid], tgt[valid].float(),
                pos_weight=pw, reduction="mean",
            )
            w = self.head_weights[head]
            total = total + w * l
            denom += w

        if denom == 0.0:
            return total  # zero tensor on right device
        return total / denom


class MultiHeadLovaszWithIgnore(nn.Module):
    """Binary Lovasz-hinge per head, with ignore_index masking.

    Useful for sparse classes (road, building) where pure BCE under-penalises
    misses on a tiny positive region. Pair with `MultiHeadBCEWithIgnore` via
    separate `loss:` entries with their own weights.
    """
    def __init__(
        self,
        head_names: List[str],
        ignore_index: int = 255,
        head_weights: Dict[str, float] | None = None,
        **kwargs,
    ):
        super().__init__()
        # Local import: keep top-level import light if SMP isn't installed elsewhere.
        from segmentation_models_pytorch.losses import LovaszLoss
        self.loss = LovaszLoss(mode="binary", from_logits=True, ignore_index=ignore_index)
        self.head_names = list(head_names)
        self.ignore_index = int(ignore_index)
        self.head_weights = {h: float(head_weights.get(h, 1.0)) if head_weights else 1.0
                             for h in self.head_names}

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
            valid = tgt != self.ignore_index
            if not bool(valid.any()):
                continue
            l = self.loss(logits, tgt)
            w = self.head_weights[head]
            total = total + w * l
            denom += w
        if denom == 0.0:
            return total
        return total / denom
