"""Trainer subclass that understands dict-shaped predictions and labels.

Subclasses `baseline.Trainer` and overrides:
  - set_to_device       (handles dict labels)
  - one_step_train      (no `.float()` on dict; loss is dict-aware)
  - one_step_validate   (same)

Everything else (validate(), checkpointing, keep-best, mlops logging skeleton)
is reused. Logger image/data hooks are skipped because the default
TensorboardLogger expects tensor labels, not dicts.
"""
from __future__ import annotations

from typing import Dict

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

    def one_step_train(self, source, labels):
        def closure(**kwargs):
            self._optimizer.zero_grad(set_to_none=True)
            predictions = self._net(source)
            losses_values = {
                name: (loss[0](predictions, labels), loss[1])
                for name, loss in self._losses.items()
            }
            total_loss = torch.sum(torch.stack([ls[0] * ls[1] for ls in losses_values.values()]))
            total_loss.backward()
            self.losses_without_weights = {m: v[0] for m, v in losses_values.items()}
            try:
                self._logger.log_train_losses(self._global_step, self.losses_without_weights)
            except Exception:
                pass
            return total_loss

        if self._optimizer_uses_closure():
            self._optimizer.step(closure)
        else:
            _ = closure()
            self._optimizer.step()

    def one_step_validate(self, source, labels):
        predictions = self._net(source)
        losses_values = {
            name: (loss[0](predictions, labels), loss[1])
            for name, loss in self._losses.items()
        }
        try:
            self._logger.log_test_losses(self._global_step,
                                         {m: v[0] for m, v in losses_values.items()})
        except Exception:
            pass
        return predictions, losses_values
