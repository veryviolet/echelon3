"""Regression 0.7.2: the PDEATHSIG worker_init_fn must be picklable, otherwise
DataLoader(multiprocessing_context='spawn') fails when pickling the worker
initializer (a closure is not picklable). We keep it a module-level function +
functools.partial.

Also: the worker_init caps intra-op threads to avoid CPU oversubscription (each worker
otherwise spins a core-count-sized cv2/torch pool → N workers ~N×cores threads, GPU starved).
"""
import pickle
import signal

import torch
from omegaconf import OmegaConf
from torch.utils.data import TensorDataset

from echelon3.creator import (
    _worker_init_fn, _pdeathsig_worker_init, create_dataloaders, create_single_dataloader,
)


def test_worker_init_fn_is_picklable():
    pickle.loads(pickle.dumps(_worker_init_fn(None)))


def _noop_user_init(worker_id):
    pass


def test_worker_init_fn_composes_picklable_user_fn():
    pickle.loads(pickle.dumps(_worker_init_fn(_noop_user_init)))


def test_worker_init_caps_intra_op_threads():
    # Runs in-process here, so snapshot/restore the global torch thread count and SIGINT
    # (the worker init also sets SIGINT=SIG_IGN).
    prev_threads = torch.get_num_threads()
    prev_sigint = signal.getsignal(signal.SIGINT)
    try:
        _pdeathsig_worker_init(0, _user_fn=None, _threads=1)
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(prev_threads)
        signal.signal(signal.SIGINT, prev_sigint)


def _cfg(train_workers, threads_per_worker=None):
    train = {"module": "torch.utils.data", "type": "DataLoader",
             "config": {"batch_size": 4, "num_workers": train_workers}}
    if threads_per_worker is not None:
        train["config"]["threads_per_worker"] = threads_per_worker
    test = {"module": "torch.utils.data", "type": "DataLoader", "config": {"batch_size": 4, "num_workers": 0}}
    return OmegaConf.create({"train": train, "test": test})


def _ds():
    return TensorDataset(torch.zeros(8, 2))


def test_threads_per_worker_not_passed_to_dataloader():
    # threads_per_worker is NOT a DataLoader kwarg — it must be popped, else construction
    # raises TypeError. num_workers=0 so no worker_init is attached; building is the assertion.
    tr, _ = create_dataloaders(_cfg(0, threads_per_worker=2), _ds(), _ds())
    assert tr.num_workers == 0


def test_threads_per_worker_reaches_worker_init():
    tr, _ = create_dataloaders(_cfg(2, threads_per_worker=3), _ds(), _ds())
    assert tr.worker_init_fn.keywords["_threads"] == 3     # carried into the partial


def test_threads_per_worker_defaults_to_one():
    tr, _ = create_dataloaders(_cfg(2), _ds(), _ds())
    assert tr.worker_init_fn.keywords["_threads"] == 1


def _single_cfg(num_workers, threads_per_worker=None):
    cfg = {"module": "torch.utils.data", "type": "DataLoader",
           "config": {"batch_size": 4, "num_workers": num_workers}}
    if threads_per_worker is not None:
        cfg["config"]["threads_per_worker"] = threads_per_worker
    return OmegaConf.create(cfg)


def test_evaluate_path_pops_threads_per_worker():
    # `evaluate` builds loaders via create_single_dataloader — it must pop threads_per_worker
    # too, else the documented dataloaders.test.config.threads_per_worker crashes evaluate.
    loader = create_single_dataloader(_single_cfg(0, threads_per_worker=1), _ds())
    assert loader.num_workers == 0                          # built without TypeError


def test_evaluate_path_applies_thread_cap():
    loader = create_single_dataloader(_single_cfg(2, threads_per_worker=3), _ds())
    assert loader.worker_init_fn.keywords["_threads"] == 3
