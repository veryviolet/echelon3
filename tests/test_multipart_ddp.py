"""MultiPartDataset под DDP:
- MultiPartBatchSampler шардит индексы максимальной части по рангам (disjoint, полное покрытие),
  per-rank __len__; на 1 GPU поведение прежнее. Проверяем явными num_replicas/rank (без реального DDP).
- create_dataloaders: MultiPart + обычный DataLoader -> понятная ошибка (а не 'int' object subscriptable);
  MultiPartDataLoader -> ок.
"""
import torch
import pytest
from omegaconf import OmegaConf
from torch.utils.data import TensorDataset

from echelon3.data.basic import MultiPartDataset
from echelon3.dataloaders.multipart import MultiPartBatchSampler
from echelon3.creator import create_dataloaders


def _fake_multipart(lens, shares):
    mp = MultiPartDataset.__new__(MultiPartDataset)   # минуем тяжёлый __init__ (create_single_dataset)
    mp.parts = [(sh, TensorDataset(torch.zeros(n, 2))) for n, sh in zip(lens, shares)]
    return mp


def test_batch_sampler_ddp_shards_max_part():
    mp = _fake_multipart([100, 40], [0.6, 0.4])          # max part = 0 (len 100)
    s0 = MultiPartBatchSampler(mp, batch_size=10, drop_last=False, num_replicas=2, rank=0)
    s1 = MultiPartBatchSampler(mp, batch_size=10, drop_last=False, num_replicas=2, rank=1)

    i0, i1 = set(s0._max_part_indices()), set(s1._max_part_indices())
    assert i0.isdisjoint(i1) and (i0 | i1) == set(range(100))   # disjoint + полное покрытие
    assert len(i0) == 50 and len(i1) == 50
    assert len(s0) == 9 and len(s1) == 9                 # ceil(50 / int(0.6*10)=6)

    # батчи — списки (part, sample)-кортежей; максимальная часть только из шарда ранга
    batches = list(iter(s0))
    assert batches and all(isinstance(b, list) and all(isinstance(x, tuple) for x in b) for b in batches)
    assert {x[1] for b in batches for x in b if x[0] == 0} <= i0


def test_batch_sampler_ddp_equal_counts_when_uneven():
    """max_part_len не делится на world -> паддинг выравнивает per-rank счётчики (иначе
    разное число батчей по рангам -> рассинхрон all-reduce -> DDP-hang)."""
    mp = _fake_multipart([101, 40], [0.6, 0.4])          # 101 % 2 != 0
    s0 = MultiPartBatchSampler(mp, batch_size=10, drop_last=False, num_replicas=2, rank=0)
    s1 = MultiPartBatchSampler(mp, batch_size=10, drop_last=False, num_replicas=2, rank=1)
    a, b = s0._max_part_indices(), s1._max_part_indices()
    assert len(a) == len(b) == 51                        # выровнено (ceil(101/2))
    assert len(s0) == len(s1)                            # равное число батчей — нет hang
    assert set(a) | set(b) == set(range(101))            # полное покрытие


def test_batch_sampler_single_gpu_unchanged():
    mp = _fake_multipart([100, 40], [0.6, 0.4])
    s = MultiPartBatchSampler(mp, batch_size=10, drop_last=False, num_replicas=1, rank=0)
    assert len(s._max_part_indices()) == 100
    assert len(s) == 17                                  # ceil(100 / 6)


def test_len_matches_iteration_when_last_part_is_largest():
    """__len__ == реальное число батчей __iter__ (оба через quants[max_part]). Иначе при
    'крупнейшая часть последняя + недробные доли' len завышался -> total_batches врал ->
    end-of-epoch валидация/чекпоинт молча пропускались."""
    mp = _fake_multipart([40, 100], [0.33, 0.67])       # часть 1 (последняя) — крупнейшая
    s = MultiPartBatchSampler(mp, batch_size=10, drop_last=False, num_replicas=1, rank=0)
    assert len(s) == len(list(iter(s)))


def test_multipart_test_slot_plain_dataloader_also_raises():
    """Понятная ошибка и для MultiPartDataset в TEST-слоте с обычным DataLoader (не только train)."""
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
    next(iter(train_dl))                                 # итерируется без int-краша
