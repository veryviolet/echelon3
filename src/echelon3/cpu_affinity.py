"""Opt-in DataLoader placement, independent of the rank's GPU-local affinity."""
from contextvars import ContextVar
from collections.abc import Sequence
from functools import wraps
import os


_launcher_cpus = ContextVar('echelon3_launcher_cpus', default=None)


def launcher_cpus():
    """Return the entrypoint snapshot, or the caller's mask for direct Python use."""
    saved = _launcher_cpus.get()
    if saved is not None:
        return saved
    if not hasattr(os, 'sched_getaffinity'):
        return None
    return tuple(sorted(os.sched_getaffinity(0)))


def capture_launcher_affinity(fn):
    """Capture before CUDA/DDP setup; nested entrypoints retain the launcher mask."""
    @wraps(fn)
    def wrapped(*args, **kwargs):
        return run_with_launcher_affinity(fn, args, kwargs, launcher_cpus())
    return wrapped


def run_with_launcher_affinity(fn, args, kwargs, cpus):
    """Module-level elastic entrypoint: carry the parent mask across spawn."""
    token = _launcher_cpus.set(cpus)
    try:
        return fn(*args, **kwargs)
    finally:
        _launcher_cpus.reset(token)


def resolve_worker_affinity(value):
    if value is None:
        return None
    if isinstance(value, str) and value == 'launcher':
        requested = None
    elif (isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and value
          and all(type(cpu) is int and cpu >= 0 for cpu in value)):
        requested = set(value)
    else:
        raise ValueError('worker_cpu_affinity must be launcher or a non-empty list of CPU IDs')
    allowed = launcher_cpus()
    if allowed is None or not hasattr(os, 'sched_setaffinity'):
        raise ValueError('worker_cpu_affinity requires Linux CPU affinity support')
    allowed = set(allowed)
    requested = allowed if requested is None else requested
    if not requested or not requested <= allowed:
        raise ValueError(
            f'worker_cpu_affinity CPUs {sorted(requested - allowed)} are outside '
            f'the launcher mask {sorted(allowed)}'
        )
    return tuple(sorted(requested))


def apply_worker_affinity(cpus):
    if cpus is None:
        return
    # The kernel enforces cpuset/cgroup restrictions. It can silently intersect
    # the mask if restrictions changed since launch: reject that partial result.
    try:
        os.sched_setaffinity(0, cpus)
        actual = os.sched_getaffinity(0)
    except OSError as exc:
        raise RuntimeError(f'Cannot apply worker_cpu_affinity {list(cpus)}') from exc
    if actual != set(cpus):
        raise RuntimeError(
            f'worker_cpu_affinity requested {list(cpus)}, got {sorted(actual)}; '
            'check current cpuset/cgroup restrictions'
        )
