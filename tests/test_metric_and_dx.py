"""Reported echelon3 shortcomings:
 1. Metric derives reset()/to()/dist_reduce() from a declared `_counters` list.
 2. Metric contract is validated at build time (create_metrics), not on the first validation.
 3. `_record_ddp_errors` wraps the rank entrypoint so a crash surfaces the real traceback.
 4. `python -m echelon3.cli` prints a helpful message.
"""
import subprocess
import sys

import pytest
import torch

from echelon3.metrics.base import Metric
from echelon3.creator import _validate_metric


class _CounterMetric(Metric):
    _counters = ("total", "count")            # reset/to/dist_reduce derived from this

    def __init__(self):
        self.reset()

    def update(self, predicted, target):
        self.total = self.total + predicted.sum()
        self.count = self.count + predicted.numel()

    def compute(self):
        return (self.total / self.count.clamp(min=1)).item()


# ---- 1. declarative counters ----

def test_counters_reset_creates_and_zeros():
    m = _CounterMetric()
    assert float(m.total) == 0.0 and float(m.count) == 0.0     # base reset() created them
    m.update(torch.ones(4), None)
    assert m.compute() == 1.0
    m.reset()
    assert float(m.total) == 0.0                                # base reset() zeroed


def test_counters_to_moves_them():
    m = _CounterMetric()
    m.to(torch.float32)                                        # base to() moves counters
    assert m.total.dtype == torch.float32


def test_counters_dist_reduce_noop_off_ddp():
    m = _CounterMetric()
    m.update(torch.ones(3), None)
    m.dist_reduce()                                           # off-DDP: no-op, no crash
    assert float(m.count) == 3.0


# ---- 2. build-time metric validation ----

def test_validate_metric_missing_method():
    class _NoReset:
        def update(self, p, t): pass
        def compute(self): return 0
        def to(self, *a, **k): return self

    with pytest.raises(TypeError, match="reset"):
        _validate_metric("m", _NoReset())


def test_validate_metric_reset_raises():
    class _BadReset(Metric):
        def update(self, p, t): pass
        def compute(self): return 0
        def reset(self): raise RuntimeError("boom")

    with pytest.raises(TypeError, match="reset"):
        _validate_metric("m", _BadReset())


def test_validate_metric_declared_counter_not_created():
    # The reported failure mode: reset() overridden and forgets a declared counter, so it only
    # blows up later in dist_reduce()/compute(). Build-time validation must catch it now.
    class _ForgotCounter(Metric):
        _counters = ("total", "count")

        def reset(self):
            self.total = torch.zeros(())      # forgot 'count'

        def update(self, p, t):
            pass

        def compute(self):
            return 0

    with pytest.raises(TypeError, match="count"):
        _validate_metric("m", _ForgotCounter())


def test_validate_metric_ok():
    _validate_metric("m", _CounterMetric())                   # valid → no raise


# ---- 3. DDP error-surfacing wrapper ----

def test_record_ddp_errors_wraps_callable():
    from echelon3.cli.train import _record_ddp_errors
    g = _record_ddp_errors(lambda x: x * 2)
    assert g(3) == 6                                          # wrapped or passthrough, still works


# ---- 4. python -m echelon3.cli message ----

def test_cli_package_run_message():
    r = subprocess.run([sys.executable, "-m", "echelon3.cli"], capture_output=True, text=True)
    assert r.returncode == 2
    assert "echelon3 train" in r.stderr
