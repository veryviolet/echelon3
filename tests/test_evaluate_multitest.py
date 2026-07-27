"""`echelon3 evaluate` used to treat data.test as a single dataset (create_single_dataset
reads .module/.type), so a multi-test config that TRAINS fine — data.test as a named dict
{test: ..., test_ms: ...} — crashed with an opaque `Missing key module`. `_resolve_test_sets`
now accepts both formats (like `train`) and gives a clear error when a named set lacks its
dataloader.
"""
import pytest
from omegaconf import OmegaConf

from echelon3.cli.evaluate import _resolve_test_sets


def _single():
    data = OmegaConf.create({"module": "my.pkg", "type": "MyDS", "config": {"a": 1}})
    loaders = OmegaConf.create({"module": "torch.utils.data", "type": "DataLoader", "config": {"batch_size": 4}})
    return data, loaders


def _named():
    data = OmegaConf.create({
        "test":    {"module": "my.pkg", "type": "MyDS",   "config": {"split": "test"}},
        "test_ms": {"module": "my.pkg", "type": "MyDSms", "config": {"split": "ms"}},
    })
    loaders = OmegaConf.create({
        "test":    {"module": "torch.utils.data", "type": "DataLoader", "config": {"batch_size": 4}},
        "test_ms": {"module": "torch.utils.data", "type": "DataLoader", "config": {"batch_size": 2}},
    })
    return data, loaders


def test_single_test_set():
    data, loaders = _single()
    sets = _resolve_test_sets(data, loaders)
    assert len(sets) == 1
    name, dcfg, lcfg = sets[0]
    assert name == "test" and dcfg is data and lcfg is loaders


def test_named_test_sets_map_to_their_loaders():
    data, loaders = _named()
    sets = _resolve_test_sets(data, loaders)
    assert [s[0] for s in sets] == ["test", "test_ms"]
    # each named set is paired with its own dataloader config
    assert sets[0][1] is data["test"] and sets[0][2] is loaders["test"]
    assert sets[1][1] is data["test_ms"] and sets[1][2] is loaders["test_ms"]


def test_named_test_set_missing_loader_raises_clear_error():
    data, loaders = _named()
    del loaders["test_ms"]                    # loader missing for one named set
    with pytest.raises(RuntimeError, match="test_ms"):
        _resolve_test_sets(data, loaders)


def test_empty_data_test_raises():
    with pytest.raises(RuntimeError, match="empty"):
        _resolve_test_sets(OmegaConf.create({}), OmegaConf.create({}))
