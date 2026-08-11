"""keep_best_on schema (0.11.0). Direction is per-name: a LOSS defaults to `low` (minimise),
a metric to `high` (maximise); override with `direction: high|low`. `value` and the global
`high_is_better` are removed (migration errors). A key that is neither a metric nor a loss is
never computed → hard error (it would silently save only the initial checkpoint). tolerance
mode uses `tolerance_value`.
"""
import tempfile

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from echelon3.trainers.baseline import Trainer
from echelon3.metrics.base import Metric
from echelon3.checkpoint.manager import CheckpointManager


class _DummyMetric(Metric):
    def update(self, predicted, target):
        pass

    def compute(self):
        return 0.0


def _trainer(loss_names=("total",), metric_names=(), keep_best_on=None, **kwargs):
    net = torch.nn.Linear(4, 4)
    loader = DataLoader(TensorDataset(torch.zeros(8, 4), torch.zeros(8, 4)), batch_size=4)
    losses = {k: (torch.nn.L1Loss(), 1.0) for k in loss_names}
    metrics = {k: _DummyMetric() for k in metric_names}
    return Trainer(
        epochs=1, train_dataloader=loader, test_dataloader=None, net=net,
        losses=losses, metrics=metrics, optimizer=torch.optim.SGD(net.parameters(), 0.01),
        scheduler=None, ckpt_manager=CheckpointManager(path=tempfile.mkdtemp(), checkpoints_to_keep=2),
        mlops_logger=None, device=torch.device("cpu"), keep_best_on=keep_best_on, **kwargs,
    )


# ---- direction defaults ----

def test_bare_loss_minimises():
    tr = _trainer(loss_names=("total",), keep_best_on="total")
    assert tr._keep_best_config["total"]["direction"] == "low"
    assert tr._metric_condition("total", 0.52, 1.03) is True      # loss fell → improved (the bug)
    assert tr._metric_condition("total", 1.50, 1.03) is False


def test_bare_metric_maximises():
    tr = _trainer(metric_names=("accuracy",), keep_best_on="accuracy")
    assert tr._keep_best_config["accuracy"]["direction"] == "high"
    assert tr._metric_condition("accuracy", 0.9, 0.8) is True


def test_explicit_direction_low_for_metric():
    tr = _trainer(metric_names=("err",), keep_best_on={"err": {"direction": "low"}})
    assert tr._keep_best_config["err"]["direction"] == "low"
    assert tr._metric_condition("err", 0.1, 0.2) is True


def test_shorthand_value_is_direction():
    tr = _trainer(metric_names=("m",), keep_best_on={"m": "low"})
    assert tr._keep_best_config["m"]["direction"] == "low"


def test_tolerance_mode():
    tr = _trainer(metric_names=("m",),
                  keep_best_on={"m": {"mode": "tolerance", "direction": "high", "tolerance_value": "0.1%"}})
    cfg = tr._keep_best_config["m"]
    assert cfg["mode"] == "tolerance" and cfg["direction"] == "high" and cfg["tolerance_value"] == "0.1%"
    assert tr._metric_condition("m", 0.9995, 1.0) is True         # 0.05% worse, within 0.1% → kept
    assert tr._metric_condition("m", 0.9, 1.0) is False           # 10% worse → not kept
    assert tr._metric_condition("m", 1.2, 1.0) is True            # better (higher) → kept


# ---- validation / migration errors ----

def test_unknown_key_is_hard_error():
    with pytest.raises(ValueError, match="neither a metric nor a loss"):
        _trainer(loss_names=("total",), keep_best_on="ghost")


def test_unknown_key_in_mixed_list_errors():
    # the reported scenario: a base recipe's keep_best_on leaks a key this run never computes
    with pytest.raises(ValueError, match="vo_score5"):
        _trainer(metric_names=("vo_score",), keep_best_on=["vo_score", "vo_score5"])


def test_legacy_value_key_errors():
    with pytest.raises(ValueError, match="value"):
        _trainer(metric_names=("m",), keep_best_on={"m": {"value": "high"}})


def test_removed_high_is_better_errors():
    with pytest.raises(ValueError, match="high_is_better"):
        _trainer(loss_names=("total",), keep_best_on="total", high_is_better=True)


def test_bad_direction_errors():
    with pytest.raises(ValueError, match="direction"):
        _trainer(metric_names=("m",), keep_best_on={"m": {"direction": "up"}})


def test_tolerance_without_value_errors():
    with pytest.raises(ValueError, match="tolerance_value"):
        _trainer(metric_names=("m",), keep_best_on={"m": {"mode": "tolerance"}})


# ---- end-to-end: falling loss saves a new checkpoint (the reported symptom) ----

class _ScriptedLoss(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = list(values)
        self.i = 0

    def forward(self, predicted, target):
        v = self.values[min(self.i, len(self.values) - 1)]
        self.i += 1
        return torch.tensor(float(v))


def _end2end(tmp, values):
    net = torch.nn.Linear(4, 4)
    train_loader = DataLoader(TensorDataset(torch.zeros(4, 4), torch.zeros(4, 4)), batch_size=4)
    test_loader = DataLoader(TensorDataset(torch.zeros(4, 4), torch.zeros(4, 4)), batch_size=4)
    tr = Trainer(
        epochs=1, train_dataloader=train_loader, test_dataloader={"test": test_loader}, net=net,
        losses={"total": (_ScriptedLoss(values), 1.0)}, metrics={},
        optimizer=torch.optim.SGD(net.parameters(), 0.01), scheduler=None,
        ckpt_manager=CheckpointManager(path=str(tmp), checkpoints_to_keep=10),
        mlops_logger=None, device=torch.device("cpu"), keep_best_on="total",
    )
    tr._global_step = 0
    return tr


def _n_ckpts(tmp):
    import os
    return len([f for f in os.listdir(tmp) if f.endswith(".tar")])


def test_loss_saves_on_decrease(tmp_path):
    tr = _end2end(tmp_path, values=[1.0305, 0.5187])
    tr.validate_and_check_for_saving(initial=True)
    tr.validate_and_check_for_saving()
    assert _n_ckpts(tmp_path) == 2                                # baseline + improvement


def test_loss_no_save_on_increase(tmp_path):
    tr = _end2end(tmp_path, values=[0.5187, 1.0305])
    tr.validate_and_check_for_saving(initial=True)
    tr.validate_and_check_for_saving()
    assert _n_ckpts(tmp_path) == 1                                # only baseline


def test_known_but_uncomputed_key_warns_not_silent(capsys):
    # A declared metric that is never populated in _current_metrics_all: must NOT silently
    # never-save — it warns loudly (once) so the misconfig is visible.
    tr = _trainer(metric_names=("computed", "never_run"),
                  keep_best_on=["computed", "never_run"])
    tr._current_metrics_all = {"computed": 0.9}          # 'never_run' absent
    tr._best_metrics_all = {"computed": 0.8}
    assert tr._all_metrics_improved() is False
    out = capsys.readouterr().out
    assert "never_run" in out and "NOT computed" in out
    # warn-once: a second check stays silent
    tr._all_metrics_improved()
    assert "never_run" not in capsys.readouterr().out
