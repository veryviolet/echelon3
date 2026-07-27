"""Regression: stateful losses were left on CPU. `self._losses = losses` (no `.to`) meant a
loss with a buffer/parameter — CrossEntropyLoss(weight=...), BCEWithLogitsLoss(pos_weight=...),
ArcFace — kept its state on CPU while predictions were on the device, so the forward failed
with a device mismatch. The trainer must move losses onto the device, like net and metrics.
"""
import tempfile

import torch
from torch.utils.data import DataLoader, TensorDataset

from echelon3.trainers.baseline import Trainer
from echelon3.checkpoint.manager import CheckpointManager


def _trainer(losses, device):
    net = torch.nn.Linear(4, 3)
    loader = DataLoader(TensorDataset(torch.zeros(8, 4), torch.zeros(8, dtype=torch.long)), batch_size=4)
    td = tempfile.mkdtemp()
    return Trainer(
        epochs=1, train_dataloader=loader, test_dataloader=None, net=net,
        losses=losses, metrics={}, optimizer=torch.optim.SGD(net.parameters(), 0.01),
        scheduler=None, ckpt_manager=CheckpointManager(path=td, checkpoints_to_keep=1),
        mlops_logger=None, device=device,
    )


class _SpyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("w", torch.zeros(3))
        self.to_devices = []

    def to(self, *args, **kwargs):
        self.to_devices.append(args[0] if args else kwargs.get("device"))
        return super().to(*args, **kwargs)

    def forward(self, pred, target):
        return (pred * self.w.sum()).sum()


def test_trainer_moves_stateful_loss_to_device():
    # Deterministic everywhere: the trainer must call .to(device) on the loss module.
    spy = _SpyLoss()
    dev = torch.device("cpu")
    tr = _trainer({"spy": (spy, 1.0)}, dev)
    assert tr._losses["spy"][0] is spy
    assert dev in spy.to_devices          # trainer moved it onto the device


def test_weighted_loss_buffer_on_trainer_device():
    # The loss buffer must end up on the trainer device (on GPU this is the actual fix;
    # on CPU it still asserts the invariant holds).
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ce = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.0, 3.0]))
    tr = _trainer({"ce": (ce, 1.0)}, dev)
    assert tr._losses["ce"][0].weight.device.type == dev.type
