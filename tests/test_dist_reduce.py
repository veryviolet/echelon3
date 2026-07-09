"""Хук distributed-редукции метрик (echelon3 0.7.7): база — no-op, all_reduce_sum_
безопасен вне DDP, кастомная метрика может свести буферы перед compute().
Полный многоранговый путь тут не проверить (1 процесс); проверяем контракт и
no-op-безопасность вне распределённого контекста."""
import torch

from echelon3.metrics.base import Metric, all_reduce_sum_


def test_all_reduce_sum_is_noop_without_distributed():
    t = torch.tensor([1.0, 2.0, 3.0])
    all_reduce_sum_(t)  # torch.distributed не инициализирован -> no-op
    assert torch.equal(t, torch.tensor([1.0, 2.0, 3.0]))


class _AccMetric(Metric, torch.nn.Module):
    """IoU-подобная метрика на аккумуляторах: dist_reduce = SUM-all-reduce буферов."""

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.register_buffer("inter", torch.zeros(1))
        self.register_buffer("union", torch.zeros(1))

    def update(self, predicted, target):
        self.inter += (predicted * target).sum()
        self.union += (predicted + target - predicted * target).sum()

    def dist_reduce(self):
        all_reduce_sum_(self.inter, self.union)

    def compute(self):
        return (self.inter / self.union.clamp_min(1)).item()

    def reset(self):
        self.inter.zero_()
        self.union.zero_()


def test_base_metric_dist_reduce_is_noop():
    assert _PureNoop().dist_reduce() is None


class _PureNoop(Metric):
    def update(self, p, t): pass
    def compute(self): return 0
    def reset(self): pass


def test_custom_metric_dist_reduce_runs_and_computes():
    m = _AccMetric()
    p = torch.tensor([1.0, 1.0, 0.0])
    t = torch.tensor([1.0, 0.0, 0.0])
    m.update(p, t)
    m.dist_reduce()          # вне DDP — буферы не меняются
    assert abs(m.compute() - (1.0 / 2.0)) < 1e-6


def test_multihead_binary_iou_dist_reduce_single_process():
    from echelon3.metrics.multibinary import MultiHeadBinaryIoU

    m = MultiHeadBinaryIoU(head_names=["road"])
    preds = {"road": torch.tensor([[[[2.0, -2.0], [2.0, 2.0]]]])}  # (1,1,2,2) logits
    labels = {"road": torch.tensor([[[1, 0], [1, 0]]])}            # (1,2,2)
    m.update(preds, labels)
    iou_before = m.compute()
    m.dist_reduce()  # вне DDP — no-op, счётчики не меняются
    assert m.compute() == iou_before
    assert abs(iou_before - 2.0 / 3.0) < 1e-6  # tp=2, fp=1, fn=0
