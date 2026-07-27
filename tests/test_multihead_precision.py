"""MultiHeadTrainer must not lose mixed precision. It used to override one_step_train /
one_step_validate and drop the base's `torch.autocast` context, so a run configured for
bf16/fp16 silently ran in fp32 (banner said AMP was on) — double the memory, half the
throughput. The fix: override `compute_losses` (the seam), inheriting the base step's
autocast + GradScaler. Also covers the validation progress unit (#3: batches, not samples).
"""
import tempfile

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from echelon3.trainers.baseline import Trainer
from echelon3.trainers.multihead import MultiHeadTrainer
from echelon3.metrics.base import Metric
from echelon3.checkpoint.manager import CheckpointManager


class _DictNet(torch.nn.Module):
    def __init__(self, seen):
        super().__init__()
        self.lin = torch.nn.Linear(4, 3)
        self._seen = seen

    def forward(self, x):
        self._seen.append(torch.is_autocast_enabled())   # was autocast active during forward?
        return {"out": self.lin(x)}


class _DictLoss(torch.nn.Module):
    def forward(self, predictions, labels):
        return torch.nn.functional.mse_loss(predictions["out"], labels["out"])


class _CountMetric(Metric):
    def __init__(self):
        self.n = 0

    def reset(self):
        self.n = 0

    def update(self, predicted, target):
        self.n += 1

    def compute(self):
        return float(self.n)


class _DictDS(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, i):
        return torch.zeros(4), {"out": torch.zeros(3)}


def _multihead(net, device, precision="bf16", test_loader=None, metrics=None):
    loader = DataLoader(_DictDS(), batch_size=4)
    return MultiHeadTrainer(
        epochs=1, train_dataloader=loader, test_dataloader=test_loader, net=net,
        losses={"mse": (_DictLoss(), 1.0)}, metrics=metrics or {},
        optimizer=torch.optim.SGD(net.parameters(), 0.01), scheduler=None,
        ckpt_manager=CheckpointManager(path=tempfile.mkdtemp(), checkpoints_to_keep=1),
        mlops_logger=None, device=device, precision=precision,
    )


def test_multihead_inherits_step_methods():
    # Structural (device-independent): overriding compute_losses — NOT the whole step — is
    # what keeps the base autocast/scaler path. If these ever diverge, precision breaks again.
    assert MultiHeadTrainer.one_step_train is Trainer.one_step_train
    assert MultiHeadTrainer.one_step_validate is Trainer.one_step_validate


@pytest.mark.skipif(not torch.cuda.is_available(), reason="autocast('cuda') needs a GPU")
def test_multihead_forward_under_autocast_when_amp_on():
    seen = []
    tr = _multihead(_DictNet(seen), torch.device("cuda"), precision="bf16")
    src, lbl = next(iter(tr._train_loader))
    src, lbl = tr.set_to_device(src, lbl)
    tr.one_step_train(src, lbl)
    assert seen and all(seen)          # forward ran inside autocast — the reported bug is fixed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="autocast('cuda') needs a GPU")
def test_multihead_no_autocast_when_fp32():
    seen = []
    tr = _multihead(_DictNet(seen), torch.device("cuda"), precision="fp32")
    src, lbl = next(iter(tr._train_loader))
    src, lbl = tr.set_to_device(src, lbl)
    tr.one_step_train(src, lbl)
    assert seen and not any(seen)      # fp32 -> autocast off


def test_validation_processes_all_batches():
    # #3: the bar unit was samples but the step fell back to 1 for non-tensor sources, so it
    # *looked* truncated. The data was always fully read — assert every batch is processed.
    cnt = _CountMetric()
    loader = DataLoader(_DictDS(), batch_size=4)    # 8 / 4 = 2 batches
    tr = _multihead(_DictNet([]), torch.device("cpu"),
                    test_loader={"test": loader}, metrics={"cnt": cnt})
    tr.validate()
    assert tr._metrics["cnt"].n == 2                # all batches seen, no truncation


def test_loader_num_batches():
    assert Trainer._loader_num_batches(DataLoader(_DictDS(), batch_size=4)) == 2
    assert Trainer._loader_num_batches(DataLoader(_DictDS(), batch_size=3)) == 3   # ceil(8/3)
