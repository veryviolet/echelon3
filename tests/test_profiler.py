"""Opt-in torch.profiler hook (trainer.config.profile). It profiles a short window
(wait/warmup/active) to quantify compute vs comm vs dataloader wait — e.g. DDP scaling on
no-NVLink hardware — writes a Chrome/TensorBoard trace, and (stop_after) stops the throwaway
run once the window is done. Off by default (no `profile` block) — a no-op.
"""
import tempfile

import torch
from torch.utils.data import DataLoader, TensorDataset

from echelon3.trainers.baseline import Trainer
from echelon3.checkpoint.manager import CheckpointManager


def _trainer(tmp, profile=None, epochs=5):
    net = torch.nn.Linear(4, 4)
    loader = DataLoader(TensorDataset(torch.zeros(16, 4), torch.zeros(16, 4)), batch_size=2)  # 8/epoch
    kw = {"profile": profile} if profile is not None else {}
    return Trainer(
        epochs=epochs, train_dataloader=loader, test_dataloader=None, net=net,
        losses={"l1": (torch.nn.L1Loss(), 1.0)}, metrics={},
        optimizer=torch.optim.SGD(net.parameters(), 0.01), scheduler=None,
        ckpt_manager=CheckpointManager(path=str(tmp), checkpoints_to_keep=2),
        mlops_logger=None, device=torch.device("cpu"), times_to_validate_per_epoch=1, **kw,
    )


def test_profiler_off_by_default(tmp_path):
    tr = _trainer(tmp_path)
    assert tr._profiler is None and tr._profile_on is False
    tr.train()                              # runs normally, no profiler, no trace dir
    assert tr._profile_stop_requested is False


def test_profiler_writes_trace_and_stops_run(tmp_path):
    outdir = tmp_path / "prof"
    tr = _trainer(tmp_path, profile={
        "wait": 1, "warmup": 1, "active": 2, "repeat": 1,
        "dir": str(outdir), "stop_after": True,
    }, epochs=5)
    tr.train()

    # window = wait+warmup+active = 4 steps → run stops there (well before 5 epochs × 8 batches)
    assert tr._profile_stop_requested is True
    assert tr._profile_steps_done >= 4
    assert tr._current_epoch == 1                       # stopped inside the first epoch
    traces = list(outdir.glob("*"))
    assert traces, "profiler wrote no trace file"       # a Chrome/TensorBoard trace was written
