"""Регрессия: echelon3.metrics.base.Metric.to() был no-op и по MRO затирал
nn.Module.to у метрик-модулей с буферами → буферы оставались на CPU и validate на
cuda падал. Metric.to должен делегировать в nn.Module.to, если метрика — модуль."""
import torch

from echelon3.metrics.base import Metric


class _BufferMetric(Metric, torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.register_buffer("state", torch.zeros(3))

    def update(self, predicted, target):
        pass

    def compute(self):
        return self.state.sum()

    def reset(self):
        self.state.zero_()


class _PureMetric(Metric):
    def update(self, predicted, target):
        pass

    def compute(self):
        return 0

    def reset(self):
        pass


def test_metric_to_moves_module_buffers():
    m = _BufferMetric()
    assert m.to(torch.device("cpu")) is m
    # dtype-перенос идёт через nn.Module.to — до фикса no-op его бы проглотил
    m.to(torch.float64)
    assert m.state.dtype == torch.float64


def test_pure_metric_to_is_noop():
    p = _PureMetric()
    assert p.to(torch.device("cpu")) is p
