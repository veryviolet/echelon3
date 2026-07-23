"""Per-head binary IoU + macro mean.

Expects:
  predictions: dict[head_name -> (B, 1, H, W) logits]   # threshold = 0
  labels:      dict[head_name -> (B, H, W) long, values in {0, 1, ignore_index}]

`.compute()` returns the macro-mean IoU across heads with at least one valid
sample seen so far. Per-head IoUs are available via `.compute_per_head()`.

A torchmetrics-compatible `to(device)` is provided so the Trainer can move the
metric onto the device just like other metrics.
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from echelon3.metrics.base import Metric, all_reduce_sum_


class MultiHeadBinaryIoU(Metric, nn.Module):
    def __init__(
        self,
        head_names: List[str],
        ignore_index: int = 255,
        threshold: float = 0.0,        # logits threshold (== prob 0.5)
        **kwargs,
    ):
        nn.Module.__init__(self)
        Metric.__init__(self)
        self.head_names = list(head_names)
        self.ignore_index = int(ignore_index)
        self.threshold = float(threshold)
        for h in self.head_names:
            self.register_buffer(f"tp_{h}", torch.zeros(()), persistent=False)
            self.register_buffer(f"fp_{h}", torch.zeros(()), persistent=False)
            self.register_buffer(f"fn_{h}", torch.zeros(()), persistent=False)
            self.register_buffer(f"n_{h}",  torch.zeros(()), persistent=False)

    def reset(self):
        for h in self.head_names:
            getattr(self, f"tp_{h}").zero_()
            getattr(self, f"fp_{h}").zero_()
            getattr(self, f"fn_{h}").zero_()
            getattr(self, f"n_{h}").zero_()

    def update(
        self,
        predictions: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
    ):
        for h in self.head_names:
            if h not in predictions or h not in labels:
                continue
            logits = predictions[h]
            tgt = labels[h]
            if logits.dim() == 4 and logits.size(1) == 1:
                logits = logits.squeeze(1)
            pred = (logits > self.threshold)
            valid = tgt != self.ignore_index
            if not bool(valid.any()):
                continue
            pred_v = pred[valid]
            tgt_v = tgt[valid].bool()
            tp = (pred_v & tgt_v).sum()
            fp = (pred_v & ~tgt_v).sum()
            fn = (~pred_v & tgt_v).sum()
            getattr(self, f"tp_{h}").add_(tp.float())
            getattr(self, f"fp_{h}").add_(fp.float())
            getattr(self, f"fn_{h}").add_(fn.float())
            getattr(self, f"n_{h}").add_(1.0)

    def dist_reduce(self):
        # DDP: reduce the raw counters across ranks BEFORE compute(). Validation
        # is sharded by DistributedSampler, and the base Metric.dist_reduce is a no-op,
        # so otherwise compute() would see only rank0's shard. Intersections/unions
        # are additive → a SUM all-reduce of the counters yields the EXACT global IoU
        # (averaging per-shard IoUs is not allowed). We reduce n_{h} too, otherwise the
        # set of "seen" heads would diverge across ranks and macro-mean would use a
        # different denominator. all_reduce_sum_ is a no-op outside distributed, so a
        # single GPU is unaffected.
        for h in self.head_names:
            all_reduce_sum_(
                getattr(self, f"tp_{h}"), getattr(self, f"fp_{h}"),
                getattr(self, f"fn_{h}"), getattr(self, f"n_{h}"),
            )

    def compute_per_head(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for h in self.head_names:
            n = float(getattr(self, f"n_{h}").item())
            if n == 0:
                continue
            tp = float(getattr(self, f"tp_{h}").item())
            fp = float(getattr(self, f"fp_{h}").item())
            fn = float(getattr(self, f"fn_{h}").item())
            denom = tp + fp + fn
            out[h] = (tp / denom) if denom > 0 else 0.0
        return out

    def compute(self) -> float:
        per_head = self.compute_per_head()
        if not per_head:
            return 0.0
        return sum(per_head.values()) / len(per_head)
