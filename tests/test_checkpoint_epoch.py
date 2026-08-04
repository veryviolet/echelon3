"""Checkpoint epoch/step semantics. The step-0 initial baseline used to be saved as epoch=1,
colliding with the checkpoint saved after the first real epoch, and resume restarted the
completed epoch. Now: the stored `epoch` is COMPLETED epochs (0 for the initial baseline, N
after epoch N), `global_step` is persisted, and resume from a new-format checkpoint starts the
NEXT epoch. Legacy checkpoints (no global_step) keep the old resume behavior.
"""
import tempfile

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from echelon3.trainers.baseline import Trainer
from echelon3.checkpoint.manager import (
    CheckpointManager, CHECKPOINT_MODEL_KEYWORD, CHECKPOINT_EPOCH_KEYWORD,
    CHECKPOINT_OPTIMIZER_KEYWORD, CHECKPOINT_GLOBAL_STEP_KEYWORD, CHECKPOINT_SCHEDULER_KEYWORD,
)


def _trainer(cm, epochs=1, reset=False):
    net = torch.nn.Linear(4, 4)
    loader = DataLoader(TensorDataset(torch.zeros(8, 4), torch.zeros(8, 4)), batch_size=4)  # 2 batches
    return Trainer(
        epochs=epochs, train_dataloader=loader, test_dataloader=None, net=net,
        losses={"l1": (torch.nn.L1Loss(), 1.0)}, metrics={},
        optimizer=torch.optim.SGD(net.parameters(), 0.01), scheduler=None,
        ckpt_manager=cm, mlops_logger=None, device=torch.device("cpu"),
        times_to_validate_per_epoch=1, reset=reset,
    )


def test_initial_is_epoch0_and_distinct_from_after_epoch1(tmp_path):
    cm = CheckpointManager(path=str(tmp_path), checkpoints_to_keep=10)
    _trainer(cm, epochs=1).train()

    c1 = cm.load_checkpoint(1)   # initial baseline (before any training)
    c2 = cm.load_checkpoint(2)   # after epoch 1
    assert c1[CHECKPOINT_EPOCH_KEYWORD] == 0
    assert c1[CHECKPOINT_GLOBAL_STEP_KEYWORD] == 0
    assert c2[CHECKPOINT_EPOCH_KEYWORD] == 1                 # distinct — no collision
    assert c2[CHECKPOINT_GLOBAL_STEP_KEYWORD] == 2           # 2 batches trained


def test_initial_save_is_announced(tmp_path, capsys):
    cm = CheckpointManager(path=str(tmp_path), checkpoints_to_keep=10)
    _trainer(cm, epochs=1).train()
    assert "Initial baseline (epoch 0)" in capsys.readouterr().out


def _save_ckpt(cm, epoch, with_global_step, global_step=0):
    net = torch.nn.Linear(4, 4)
    opt = torch.optim.SGD(net.parameters(), 0.01)
    ckpt = {
        CHECKPOINT_MODEL_KEYWORD: net.state_dict(),
        CHECKPOINT_EPOCH_KEYWORD: epoch,
        CHECKPOINT_OPTIMIZER_KEYWORD: opt.state_dict(),
    }
    if with_global_step:
        ckpt[CHECKPOINT_GLOBAL_STEP_KEYWORD] = global_step
    cm.save_checkpoint(ckpt)


def test_resume_new_format_starts_next_epoch(tmp_path):
    cm = CheckpointManager(path=str(tmp_path), checkpoints_to_keep=10)
    _save_ckpt(cm, epoch=3, with_global_step=True, global_step=6)   # completed 3 epochs
    tr = _trainer(cm, epochs=10)
    tr.load_from_checkpoint()
    assert tr._current_epoch == 4          # resume the NEXT epoch, not re-run 3
    assert tr._global_step == 6


def test_resume_legacy_format_unchanged(tmp_path):
    cm = CheckpointManager(path=str(tmp_path), checkpoints_to_keep=10)
    _save_ckpt(cm, epoch=3, with_global_step=False)                 # old format: no global_step
    tr = _trainer(cm, epochs=10)
    tr.load_from_checkpoint()
    assert tr._current_epoch == 3          # legacy semantics preserved


def test_resume_new_format_reconciles_scheduler(tmp_path):
    # The end-of-epoch checkpoint is saved BEFORE that epoch's scheduler.step(), so its stored
    # scheduler lags the completed-epoch count by one. Resuming at N+1 must reconcile it, else
    # every post-resume epoch runs one LR step behind.
    net = torch.nn.Linear(4, 4)
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5)
    opt.step()     # dummy (no grads) — keep the optimizer.step-before-scheduler.step order
    sched.step()   # scheduler state at end-of-epoch-2 save: last_epoch=1, lr=0.05 (epoch-2 lr)

    cm = CheckpointManager(path=str(tmp_path), checkpoints_to_keep=10)
    cm.save_checkpoint({
        CHECKPOINT_MODEL_KEYWORD: net.state_dict(),
        CHECKPOINT_EPOCH_KEYWORD: 2,                 # completed 2 epochs
        CHECKPOINT_GLOBAL_STEP_KEYWORD: 4,
        CHECKPOINT_OPTIMIZER_KEYWORD: opt.state_dict(),
        CHECKPOINT_SCHEDULER_KEYWORD: sched.state_dict(),
    })

    net2 = torch.nn.Linear(4, 4)
    opt2 = torch.optim.SGD(net2.parameters(), lr=0.1)
    sched2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=1, gamma=0.5)
    loader = DataLoader(TensorDataset(torch.zeros(8, 4), torch.zeros(8, 4)), batch_size=4)
    tr = Trainer(
        epochs=5, train_dataloader=loader, test_dataloader=None, net=net2,
        losses={"l1": (torch.nn.L1Loss(), 1.0)}, metrics={}, optimizer=opt2, scheduler=sched2,
        ckpt_manager=cm, mlops_logger=None, device=torch.device("cpu"),
        times_to_validate_per_epoch=1,
    )
    tr.load_from_checkpoint()

    assert tr._current_epoch == 3
    assert tr._scheduler.last_epoch == 2                                 # reconciled from 1
    assert abs(tr._optimizer.param_groups[0]["lr"] - 0.025) < 1e-9       # epoch-3 lr, not 0.05
