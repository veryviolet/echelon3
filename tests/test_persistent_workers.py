"""Умный дефолт persistent_workers: движок держит DataLoader-воркеров живыми между эпохами
(num_workers>0), иначе их переспавн на границе эпохи ловит Ctrl-C в момент bootstrap ->
трейсбеки + утёкшие семафоры. Guard'ы: только при num_workers>0 (иначе torch = ValueError),
и через setdefault (явное значение юзера не перетираем)."""
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
    # num_workers=0 — НЕ ставим (persistent_workers=True + 0 воркеров = ValueError у torch)
    tr, _ = create_dataloaders(_cfg(0), _ds(), _ds())
    assert tr.persistent_workers is False   # torch-дефолт, не наш


def test_persistent_workers_respects_explicit_false():
    tr, _ = create_dataloaders(_cfg(2, train_persist=False), _ds(), _ds())
    assert tr.persistent_workers is False


def test_eval_loader_not_persistent_by_default():
    # eval НЕ форсим: резидентные eval-воркеры весь прогон зря держат RAM (мотивация слабее).
    _, te = create_dataloaders(_cfg(0, test_workers=2), _ds(), _ds())
    assert te.persistent_workers is False
