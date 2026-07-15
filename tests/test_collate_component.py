"""Движок: `dataloaders.*.config.collate_fn` можно задать КОМПОНЕНТОМ (module/type/config) —
create_dataloaders строит его в callable и передаёт в DataLoader (нужно для батчинга
переменного размера — графы/сеты, напр. докинг)."""
import torch
from omegaconf import OmegaConf
from torch.utils.data import TensorDataset, default_collate

from echelon3.creator import create_dataloaders


def test_collate_fn_component_is_built():
    ds = TensorDataset(torch.zeros(8, 2), torch.zeros(8))
    collate = {"module": "torch.utils.data", "type": "default_collate", "config": {}}
    cfg = OmegaConf.create({
        "train": {"module": "torch.utils.data", "type": "DataLoader",
                  "config": {"batch_size": 4, "collate_fn": collate}},
        "test":  {"module": "torch.utils.data", "type": "DataLoader",
                  "config": {"batch_size": 4, "collate_fn": collate}},
    })
    train_dl, test_dl = create_dataloaders(cfg, ds, ds)
    assert train_dl.collate_fn is default_collate       # dict -> built callable, not left a dict
    assert callable(test_dl.collate_fn)
    next(iter(train_dl))                                 # actually batches
