"""Trainer subclass that understands dict-shaped predictions and labels.

Subclasses `baseline.Trainer` and overrides only the extension seams:
  - set_to_device          — moves dict labels onto the device;
  - compute_losses         — forward + dict-aware losses, without the tensor-only
                             `.float()` / prediction-squeeze the base applies;
  - _log_train/test_step_data — no-op, since the tensor-oriented mlops loggers can't
                             render dict labels / predictions.

Everything else — one_step_train / one_step_validate (and with them the autocast +
GradScaler + closure precision path), validate(), checkpointing, keep-best — is inherited
unchanged. Overriding `compute_losses` (the designed seam) rather than the whole step is
what keeps mixed precision working: a hand-rolled step easily drifts from the base and
silently drops autocast/scaler.
"""
from __future__ import annotations

import torch

from echelon3.trainers.baseline import Trainer


class MultiHeadTrainer(Trainer):

    def set_to_device(self, source, labels):
        source = source.to(self._device, non_blocking=True)
        if isinstance(labels, dict):
            labels = {k: v.to(self._device, non_blocking=True) for k, v in labels.items()}
        else:
            labels = labels.to(self._device, non_blocking=True)
        return source, labels

    def compute_losses(self, source, labels, net=None):
        # dict predictions/labels: pass them to the losses whole, with no `.float()` and no
        # prediction-squeeze (both assume tensor shapes). Runs inside the base's autocast
        # context; the base owns scaler / backward / closure / optimizer step.
        net = net if net is not None else self._net
        predictions = net(source)
        losses_values = {
            name: (loss[0](predictions, labels), loss[1])
            for name, loss in self._losses.items()
        }
        return predictions, losses_values

    # dict-shaped source/labels/predictions — the image/box/heatmap loggers expect tensors.
    def _log_train_step_data(self, source, labels, predictions):
        pass

    def _log_test_step_data(self, source, labels, predictions):
        pass
