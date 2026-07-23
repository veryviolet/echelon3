"""MultiDatasetMetric orchestration in Trainer.validate():
  * reset ONCE before all its datasets, update carries the source-dataset tag, compute
    ONCE after all datasets; the value is tracked by keep_best by name;
  * ordinary (single-dataset) metrics keep working per loader — regression;
  * a metric declaring a dataset absent from the test loaders is a hard error.
"""
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from echelon3.trainers.baseline import Trainer
from echelon3.metrics.base import Metric, MultiDatasetMetric
from echelon3.checkpoint.manager import CheckpointManager


class _FakeOrdinary(Metric):
    def __init__(self):
        self.update_calls = 0
        self.compute_calls = 0
        self.reset_calls = 0
        self.reset()

    def reset(self):
        self.reset_calls += 1

    def update(self, predicted, target):
        self.update_calls += 1

    def compute(self):
        self.compute_calls += 1
        return 0.5


class _FakeRetrieval(MultiDatasetMetric):
    def __init__(self, query_dataset, gallery_dataset):
        self.datasets = [query_dataset, gallery_dataset]
        self._q = query_dataset
        self.reset_calls = 0
        self.compute_calls = 0
        self.seen = []
        self.reset()

    def reset(self):
        self.reset_calls += 1
        self.seen = []

    def update(self, predicted, target, dataset):
        self.seen.append(dataset)

    def compute(self):
        self.compute_calls += 1
        return float(sum(1 for d in self.seen if d == self._q))   # # of query-side updates


def _loader(n, bs=4):
    return DataLoader(TensorDataset(torch.zeros(n, 4), torch.zeros(n, 4)), batch_size=bs)


def _trainer(tmp_path, metrics, test_dataloader, keep_best_on=None, **kwargs):
    net = torch.nn.Linear(4, 4)
    return Trainer(
        epochs=1,
        train_dataloader=_loader(8),
        test_dataloader=test_dataloader,
        net=net,
        losses={"l1": (torch.nn.L1Loss(), 1.0)},
        metrics=metrics,
        optimizer=torch.optim.SGD(net.parameters(), lr=0.01),
        scheduler=None,
        ckpt_manager=CheckpointManager(path=str(tmp_path), checkpoints_to_keep=1),
        mlops_logger=None,
        device=torch.device("cpu"),
        keep_best_on=keep_best_on,
        **kwargs,
    )


def test_multidataset_metric_reset_once_update_tagged_compute_once(tmp_path):
    retr = _FakeRetrieval("queries", "gallery")
    ordinary = _FakeOrdinary()
    trainer = _trainer(
        tmp_path,
        metrics={"ord": ordinary, "retr": retr},
        test_dataloader={"queries": _loader(8), "gallery": _loader(6)},  # 2 + 2 batches
        keep_best_on="retr",
        metrics_on={"ord": "queries"},   # ordinary routed to one loader; multi ignores routing
    )
    trainer.validate()

    # multi: reset once in validate (+1 from __init__), compute once, saw BOTH datasets
    assert retr.reset_calls == 2
    assert retr.compute_calls == 1
    assert set(retr.seen) == {"queries", "gallery"}
    # keep_best tracks it by name: value == # of query-side updates (queries loader = 2 batches)
    assert trainer._metric_to_track == 2.0

    # ordinary still computed — only on its routed loader (queries): one compute, one reset there
    assert ordinary.compute_calls == 1
    assert ordinary.update_calls == 2


def test_ordinary_metrics_still_work_per_loader(tmp_path):
    # Regression: no multi metric, no metrics_on -> ordinary metric runs on EVERY loader.
    ordinary = _FakeOrdinary()
    trainer = _trainer(
        tmp_path,
        metrics={"ord": ordinary},
        test_dataloader={"queries": _loader(8), "gallery": _loader(6)},
    )
    trainer.validate()
    assert ordinary.compute_calls == 2          # once per loader
    assert ordinary.update_calls == 4           # 2 + 2 batches


def test_missing_declared_dataset_raises(tmp_path):
    trainer = _trainer(
        tmp_path,
        metrics={"retr": _FakeRetrieval("queries", "nope")},
        test_dataloader={"queries": _loader(8)},
    )
    with pytest.raises(ValueError, match="nope"):
        trainer.validate()


def test_empty_datasets_roster_raises(tmp_path):
    # A multi-metric with an empty roster would silently update nothing and compute over
    # empty buffers — a hard error instead of a garbage value driving keep_best.
    empty = _FakeRetrieval("queries", "gallery")
    empty.datasets = []
    trainer = _trainer(
        tmp_path,
        metrics={"retr": empty},
        test_dataloader={"queries": _loader(8), "gallery": _loader(6)},
    )
    with pytest.raises(ValueError, match="no datasets"):
        trainer.validate()
