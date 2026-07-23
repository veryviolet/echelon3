"""Regression: echelon3.metrics.base.Metric.to() was a no-op and, via MRO, shadowed
nn.Module.to for metric modules with buffers → the buffers stayed on the CPU and validate on
cuda crashed. Metric.to must delegate to nn.Module.to when the metric is a module."""
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
    # the dtype transfer goes through nn.Module.to — before the fix the no-op would have swallowed it
    m.to(torch.float64)
    assert m.state.dtype == torch.float64


def test_pure_metric_to_is_noop():
    p = _PureMetric()
    assert p.to(torch.device("cpu")) is p
