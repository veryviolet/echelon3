"""Resume applies config PARTIALLY: the checkpoint's optimizer state (lr, weight_decay, …)
overrides the config, while non-optimizer config (batch_size) still applies. That silent mix
cost real runs. On resume the trainer must WARN per overridden optimizer hyperparameter and
point at trainer.config.reset: true.
"""
import tempfile

import torch
from torch.utils.data import DataLoader, TensorDataset

from echelon3.trainers.baseline import Trainer
from echelon3.checkpoint.manager import (
    CheckpointManager, CHECKPOINT_MODEL_KEYWORD, CHECKPOINT_EPOCH_KEYWORD,
    CHECKPOINT_OPTIMIZER_KEYWORD, CHECKPOINT_SCHEDULER_KEYWORD,
)


def _trainer(optimizer, ckpt_path, reset=False, scheduler=None):
    net = torch.nn.Linear(4, 3)
    loader = DataLoader(TensorDataset(torch.zeros(8, 4), torch.zeros(8, dtype=torch.long)), batch_size=4)
    return Trainer(
        epochs=1, train_dataloader=loader, test_dataloader=None, net=net,
        losses={"ce": (torch.nn.CrossEntropyLoss(), 1.0)}, metrics={},
        optimizer=optimizer, scheduler=scheduler,
        ckpt_manager=CheckpointManager(path=ckpt_path, checkpoints_to_keep=2),
        mlops_logger=None, device=torch.device("cpu"), reset=reset,
    )


def _save_scheduled_ckpt(path, base_lr, decay_steps):
    """Save a checkpoint whose optimizer lr has been DECAYED by a scheduler (base_lr stays
    in initial_lr / the scheduler's base_lrs)."""
    net = torch.nn.Linear(4, 3)
    opt = torch.optim.SGD(net.parameters(), lr=base_lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.1)
    for _ in range(decay_steps):
        opt.step()
        sched.step()
    CheckpointManager(path=path, checkpoints_to_keep=2).save_checkpoint({
        CHECKPOINT_MODEL_KEYWORD: net.state_dict(),
        CHECKPOINT_EPOCH_KEYWORD: 3,
        CHECKPOINT_OPTIMIZER_KEYWORD: opt.state_dict(),
        CHECKPOINT_SCHEDULER_KEYWORD: sched.state_dict(),
    })


def _scheduled_trainer(base_lr, ckpt_path):
    net = torch.nn.Linear(4, 3)
    opt = torch.optim.SGD(net.parameters(), lr=base_lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.1)
    return _trainer(opt, ckpt_path, reset=False, scheduler=sched)


def test_resume_warns_on_lr_override(tmp_path, capsys):
    # A checkpoint saved at lr=0.001.
    net1 = torch.nn.Linear(4, 3)
    opt1 = torch.optim.AdamW(net1.parameters(), lr=0.001)
    CheckpointManager(path=str(tmp_path), checkpoints_to_keep=2).save_checkpoint({
        CHECKPOINT_MODEL_KEYWORD: net1.state_dict(),
        CHECKPOINT_EPOCH_KEYWORD: 3,
        CHECKPOINT_OPTIMIZER_KEYWORD: opt1.state_dict(),
    })
    # Resume with a DIFFERENT config lr.
    net2 = torch.nn.Linear(4, 3)
    tr = _trainer(torch.optim.AdamW(net2.parameters(), lr=0.005), str(tmp_path), reset=False)
    tr.load_from_checkpoint()

    out = capsys.readouterr().out
    assert "WARNING" in out and "lr" in out
    assert "0.005" in out and "0.001" in out          # config -> checkpoint
    assert "reset" in out.lower()
    assert tr._optimizer.param_groups[0]["lr"] == 0.001   # checkpoint value actually won


def test_resume_no_warn_when_hparams_match(tmp_path, capsys):
    net1 = torch.nn.Linear(4, 3)
    opt1 = torch.optim.AdamW(net1.parameters(), lr=0.001)
    CheckpointManager(path=str(tmp_path), checkpoints_to_keep=2).save_checkpoint({
        CHECKPOINT_MODEL_KEYWORD: net1.state_dict(),
        CHECKPOINT_EPOCH_KEYWORD: 3,
        CHECKPOINT_OPTIMIZER_KEYWORD: opt1.state_dict(),
    })
    net2 = torch.nn.Linear(4, 3)
    tr = _trainer(torch.optim.AdamW(net2.parameters(), lr=0.001), str(tmp_path), reset=False)  # same lr
    tr.load_from_checkpoint()
    assert "WARNING" not in capsys.readouterr().out


def test_scheduled_resume_no_warn_when_base_lr_unchanged(tmp_path, capsys):
    # THE regression: with an LR scheduler the checkpoint lr is the DECAYED value. Resuming
    # with the SAME base lr must NOT warn (comparing base lr, not the live decayed lr).
    _save_scheduled_ckpt(str(tmp_path), base_lr=0.005, decay_steps=2)   # ckpt lr = 5e-5
    tr = _scheduled_trainer(base_lr=0.005, ckpt_path=str(tmp_path))
    tr.load_from_checkpoint()
    assert "WARNING" not in capsys.readouterr().out


def test_scheduled_resume_warns_on_base_lr_change(tmp_path, capsys):
    # But changing the BASE lr on a scheduled resume IS a real partial-config case — warn.
    _save_scheduled_ckpt(str(tmp_path), base_lr=0.005, decay_steps=2)
    tr = _scheduled_trainer(base_lr=0.01, ckpt_path=str(tmp_path))         # changed base lr
    tr.load_from_checkpoint()
    out = capsys.readouterr().out
    assert "WARNING" in out and "lr" in out
    assert "0.01" in out and "0.005" in out                               # config base -> ckpt base


def test_warn_method_no_false_positive_on_betas_list_tuple(tmp_path, capsys):
    # betas may round-trip tuple<->list through the checkpoint — must NOT warn on that alone.
    net = torch.nn.Linear(4, 3)
    tr = _trainer(torch.optim.AdamW(net.parameters(), lr=0.001), str(tmp_path))
    tr._warn_config_hparams_overridden([{"betas": (0.9, 0.999)}], [{"betas": [0.9, 0.999]}])
    assert "WARNING" not in capsys.readouterr().out
