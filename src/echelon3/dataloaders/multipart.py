from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, TypeVar, Generic, Sequence, List, Optional, Union, Dict, Tuple
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler
from torchvision import datasets
from torchvision.transforms import ToTensor
from omegaconf import OmegaConf, DictConfig, ListConfig
import cv2
import numpy as np
from random import shuffle as shuffle

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')
_worker_init_fn_t = Callable[[int], None]
_collate_fn_t = Callable[[List[T]], Any]

from echelon3.creator import create_universal, create_single_dataset
from echelon3.data.basic import MultiPartDataset


class MultiPartBatchSampler(BatchSampler):

    dataset = None
    batch_size = None
    drop_last = None
    quants = None
    idxs = None
    unique_batches = None

    def __init__(self, dataset: MultiPartDataset, batch_size: int, drop_last: bool,
                 num_replicas: Optional[int] = None, rank: Optional[int] = None) -> None:
        super().__init__(sampler=None, batch_size=batch_size, drop_last=drop_last)  # type: ignore[arg-type]

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        # DDP-осведомлённость: индексы максимальной части шардим по рангу (disjoint между
        # рангами). По умолчанию берём rank/world из инициализированной process group; на
        # 1 GPU (num_replicas=1) поведение прежнее.
        from echelon3 import ddp
        self.num_replicas = int(num_replicas) if num_replicas is not None else ddp.world_size()
        self.rank = int(rank) if rank is not None else ddp.rank()
        self.quants = [int(self.dataset.part_share(i) * self.batch_size) for i in range(self.dataset.parts_num())]
        self.quants[-1] = self.batch_size - sum(self.quants[:-1])

    def _max_part_indices(self):
        """Индексы максимальной части, шардированные по рангу.

        Дополняем список до кратного num_replicas (повтор с начала — как DistributedSampler),
        чтобы у ВСЕХ рангов было РОВНО одинаковое число сэмплов: иначе разное число батчей
        по рангам рассинхронит DDP-all-reduce градиентов → тихий hang. На 1 GPU — полный набор."""
        n = self.dataset.part_len(self.dataset.max_part())
        idxs = list(range(n))
        per_rank = (n + self.num_replicas - 1) // self.num_replicas
        total = per_rank * self.num_replicas
        if total > n:
            idxs = idxs + idxs[: total - n]          # паддинг повтором до кратного
        return idxs[self.rank::self.num_replicas]    # ровно per_rank у каждого ранга

    def get_part_idx(self, part) -> int:
        for i in self.idxs[part]:
            yield i

    def prepare_idxs(self):
        max_part = self.dataset.max_part()
        max_part_batch_size = self.quants[max_part]

        if max_part_batch_size <= 0:
            raise RuntimeError(
                f"MultiPartBatchSampler: max_part_batch_size <= 0 "
                f"(max_part={max_part}, quants={self.quants}). "
                f"batch_size may be too small, or shares are invalid."
            )

        # Шард максимальной части для ЭТОГО ранга; unique_batches — по per-rank длине.
        max_idxs = self._max_part_indices()
        per_rank_max_len = len(max_idxs)
        self.unique_batches = (per_rank_max_len + max_part_batch_size - 1) // max_part_batch_size

        self.idxs = [[] for _ in range(self.dataset.parts_num())]
        self.idxs[max_part] = max_idxs

        for i in range(self.dataset.parts_num()):
            if i == max_part:
                continue

            part_len = self.dataset.part_len(i)

            if part_len == 0 and self.quants[i] > 0:
                raise RuntimeError(
                    f"MultiPartBatchSampler: dataset.part_len({i}) == 0 at batch_size={self.batch_size}, "
                    f"part_share={self.dataset.part_share(i)}, quants[{i}]={self.quants[i]}. "
                    f"Check the MultiPartDataset config and the folder contents for part {i}."
                )

            part_required_size = self.unique_batches * self.quants[i]

            if part_len > 0 and part_required_size > 0:
                copies = (part_required_size + part_len - 1) // part_len
                self.idxs[i] = (list(range(part_len)) * copies)[:part_required_size]
            else:
                self.idxs[i] = []

    def __iter__(self) -> Iterator[List[int]]:
        self.prepare_idxs()
        for i in range(len(self.idxs)):
            shuffle(self.idxs[i])
        for _ in range(self.unique_batches):
            batch = []
            for p in range(self.dataset.parts_num()):
                for _ in range(self.quants[p]):
                    if len(self.idxs[p]) > 0:
                        batch.append((p, self.idxs[p].pop()))
            yield batch

    def __len__(self) -> int:
        max_part = self.dataset.max_part()
        # ВАЖНО: тот же quants[max_part], что и в __iter__/prepare_idxs (у последней части он
        # = batch_size − сумма остальных). int(share*batch_size) расходится с реальной
        # итерацией -> len завышается -> total_batches неверен -> пропуск end-of-epoch
        # валидации/чекпоинта (триггер batch+1==total_batches никогда не срабатывает).
        max_part_batch_size = self.quants[max_part]

        if max_part_batch_size <= 0:
            raise RuntimeError(
                f"MultiPartBatchSampler.__len__: max_part_batch_size <= 0 "
                f"(max_part={max_part}, share={self.dataset.part_share(max_part)}, "
                f"batch_size={self.batch_size})"
            )

        per_rank_max_len = len(self._max_part_indices())     # число батчей — на этот ранг
        if self.drop_last:
            return per_rank_max_len // max_part_batch_size
        else:
            return (per_rank_max_len + max_part_batch_size - 1) // max_part_batch_size


class MultiPartDataLoader(DataLoader):

    def __init__(self, dataset: Dataset, batch_size: Optional[int] = 1,
                 shuffle: bool = False, sampler: Union[Sampler, Iterable, None] = None,
                 batch_sampler: Union[Sampler[Sequence], Iterable[Sequence], None] = None,
                 num_workers: int = 0, collate_fn: Optional[_collate_fn_t] = None,
                 pin_memory: bool = False, drop_last: bool = False,
                 timeout: float = 0, worker_init_fn: Optional[_worker_init_fn_t] = None,
                 multiprocessing_context=None, generator=None,
                 *, prefetch_factor: Optional[int] = None,   # None: torch сам (2 при workers>0, ок при 0)
                 persistent_workers: bool = False):
        super(MultiPartDataLoader, self).__init__(
            dataset=dataset,
            batch_sampler=MultiPartBatchSampler(dataset=dataset,
                                                batch_size=batch_size,
                                                drop_last=drop_last),
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )

    def total_batches(self):
        return len(self.batch_sampler)   # type: ignore[arg-type]