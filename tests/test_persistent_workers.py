"""Smart default for persistent_workers: the engine keeps DataLoader workers alive between epochs
(num_workers>0), otherwise their respawn at the epoch boundary catches Ctrl-C during bootstrap ->
tracebacks + leaked semaphores. Guards: only when num_workers>0 (otherwise torch = ValueError),
and via setdefault (we don't overwrite the user's explicit value)."""
import torch
from omegaconf import OmegaConf
from torch.utils.data import TensorDataset

from echelon3.creator import create_dataloaders


def _cfg(train_workers, train_persist=None, test_workers=0):
    train = {"module": "torch.utils.data", "type": "DataLoader",
             "config": {"batch_size": 4, "num_workers": train_workers}}
    if train_persist is not None:
        train["config"]["persistent_workers"] = train_persist
    test = {"module": "torch.utils.data", "type": "DataLoader",
            "config": {"batch_size": 4, "num_workers": test_workers}}
    return OmegaConf.create({"train": train, "test": test})


def _ds():
    return TensorDataset(torch.zeros(8, 2))


def test_persistent_workers_default_true_when_workers():
    tr, _ = create_dataloaders(_cfg(2), _ds(), _ds())
    assert tr.persistent_workers is True


def test_persistent_workers_not_set_when_no_workers():
    # num_workers=0 — do NOT set it (persistent_workers=True + 0 workers = ValueError in torch)
    tr, _ = create_dataloaders(_cfg(0), _ds(), _ds())
    assert tr.persistent_workers is False   # torch default, not ours


def test_persistent_workers_respects_explicit_false():
    tr, _ = create_dataloaders(_cfg(2, train_persist=False), _ds(), _ds())
    assert tr.persistent_workers is False


def test_eval_loader_not_persistent_by_default():
    # we do NOT force eval: resident eval workers needlessly hold RAM for the whole run (weaker motivation).
    _, te = create_dataloaders(_cfg(0, test_workers=2), _ds(), _ds())
    assert te.persistent_workers is False
