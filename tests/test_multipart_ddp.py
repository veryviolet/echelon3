"""MultiPartDataset under DDP:
- MultiPartBatchSampler shards the indices of the largest part across ranks (disjoint, full coverage),
  per-rank __len__; on a single GPU the behavior is unchanged. We check with explicit num_replicas/rank (no real DDP).
- create_dataloaders: MultiPart + a plain DataLoader -> a clear error (not 'int' object subscriptable);
  MultiPartDataLoader -> ok.
"""
import torch
import pytest
from omegaconf import OmegaConf
from torch.utils.data import TensorDataset

from echelon3.data.basic import MultiPartDataset
from echelon3.dataloaders.multipart import MultiPartBatchSampler
from echelon3.creator import create_dataloaders


def _fake_multipart(lens, shares):
    mp = MultiPartDataset.__new__(MultiPartDataset)   # bypass the heavy __init__ (create_single_dataset)
    mp.parts = [(sh, TensorDataset(torch.zeros(n, 2))) for n, sh in zip(lens, shares)]
    return mp


def test_batch_sampler_ddp_shards_max_part():
    mp = _fake_multipart([100, 40], [0.6, 0.4])          # max part = 0 (len 100)
    s0 = MultiPartBatchSampler(mp, batch_size=10, drop_last=False, num_replicas=2, rank=0)
    s1 = MultiPartBatchSampler(mp, batch_size=10, drop_last=False, num_replicas=2, rank=1)

    i0, i1 = set(s0._max_part_indices()), set(s1._max_part_indices())
    assert i0.isdisjoint(i1) and (i0 | i1) == set(range(100))   # disjoint + full coverage
    assert len(i0) == 50 and len(i1) == 50
    assert len(s0) == 9 and len(s1) == 9                 # ceil(50 / int(0.6*10)=6)

    # batches are lists of (part, sample) tuples; the largest part comes only from this rank's shard
    batches = list(iter(s0))
    assert batches and all(isinstance(b, list) and all(isinstance(x, tuple) for x in b) for b in batches)
    assert {x[1] for b in batches for x in b if x[0] == 0} <= i0


def test_batch_sampler_ddp_equal_counts_when_uneven():
    """max_part_len not divisible by world size -> padding equalizes the per-rank counts (otherwise
    a different number of batches per rank -> all-reduce desync -> DDP hang)."""
    mp = _fake_multipart([101, 40], [0.6, 0.4])          # 101 % 2 != 0
    s0 = MultiPartBatchSampler(mp, batch_size=10, drop_last=False, num_replicas=2, rank=0)
    s1 = MultiPartBatchSampler(mp, batch_size=10, drop_last=False, num_replicas=2, rank=1)
    a, b = s0._max_part_indices(), s1._max_part_indices()
    assert len(a) == len(b) == 51                        # equalized (ceil(101/2))
    assert len(s0) == len(s1)                            # equal number of batches — no hang
    assert set(a) | set(b) == set(range(101))            # full coverage


def test_batch_sampler_single_gpu_unchanged():
    mp = _fake_multipart([100, 40], [0.6, 0.4])
    s = MultiPartBatchSampler(mp, batch_size=10, drop_last=False, num_replicas=1, rank=0)
    assert len(s._max_part_indices()) == 100
    assert len(s) == 17                                  # ceil(100 / 6)


def test_len_matches_iteration_when_last_part_is_largest():
    """__len__ == the actual number of batches from __iter__ (both via quants[max_part]). Otherwise, with
    'largest part is last + non-integer shares', len was overestimated -> total_batches lied ->
    end-of-epoch validation/checkpoint were silently skipped."""
    mp = _fake_multipart([40, 100], [0.33, 0.67])       # part 1 (the last one) is the largest
    s = MultiPartBatchSampler(mp, batch_size=10, drop_last=False, num_replicas=1, rank=0)
    assert len(s) == len(list(iter(s)))


def test_multipart_test_slot_plain_dataloader_also_raises():
    """A clear error for MultiPartDataset in the TEST slot with a plain DataLoader too (not only train)."""
    cfg = OmegaConf.create({
        "train": {"module": "echelon3.dataloaders.multipart", "type": "MultiPartDataLoader",
                  "config": {"batch_size": 4}},
        "test":  {"module": "torch.utils.data", "type": "DataLoader", "config": {"batch_size": 4}},
    })
    with pytest.raises(TypeError, match="MultiPartDataLoader"):
        create_dataloaders(cfg, _fake_multipart([20, 10], [0.6, 0.4]),
                           _fake_multipart([12, 6], [0.5, 0.5]))


def test_multipart_plain_dataloader_raises_clear_error():
    cfg = OmegaConf.create({
        "train": {"module": "torch.utils.data", "type": "DataLoader", "config": {"batch_size": 4}},
        "test":  {"module": "torch.utils.data", "type": "DataLoader", "config": {"batch_size": 4}},
    })
    with pytest.raises(TypeError, match="MultiPartDataLoader"):
        create_dataloaders(cfg, _fake_multipart([20, 10], [0.6, 0.4]), TensorDataset(torch.zeros(8, 2)))


def test_multipart_dataloader_ok():
    cfg = OmegaConf.create({
        "train": {"module": "echelon3.dataloaders.multipart", "type": "MultiPartDataLoader",
                  "config": {"batch_size": 4}},
        "test":  {"module": "torch.utils.data", "type": "DataLoader", "config": {"batch_size": 4}},
    })
    train_dl, _ = create_dataloaders(cfg, _fake_multipart([20, 10], [0.6, 0.4]),
                                     TensorDataset(torch.zeros(8, 2)))
    assert isinstance(train_dl.batch_sampler, MultiPartBatchSampler)
    next(iter(train_dl))                                 # iterates without an int crash
