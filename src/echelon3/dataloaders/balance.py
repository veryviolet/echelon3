from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, TypeVar, Generic, Sequence, List, Optional, Union, Dict, Tuple, List
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


from echelon3.data.basic import PerClassFilesDataset


class BalancedBatchSampler(BatchSampler):

    batch_size = None
    drop_last = None
    classes = None
    idxs = None
    unique_batches = None
    dataset = None
    classes_items = None

    def __init__(self, dataset: PerClassFilesDataset, classes: int, batch_size: int, drop_last: bool,
                 ) -> None:
        super().__init__(sampler=None, batch_size=batch_size, drop_last=drop_last)  # type: ignore[arg-type]

        self.dataset = dataset
        self.classes = classes
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.quants = [self.batch_size // self.classes] * self.classes
        self.quants[-1] = self.batch_size - sum(self.quants[:-1])

    def get_part_idx(self, part) -> int:
        for i in self.idxs[part]:
            yield i

    def prepare_idxs(self):
        self.classes_items = self.dataset.labels_stats()
        f = lambda i: self.classes_items[i]
        largest_class = max(range(self.classes), key=f)

        self.unique_batches = (self.classes_items[largest_class] +
                               self.quants[largest_class] - 1) // self.quants[largest_class]

        self.idxs = [[] for i in range(self.classes)]

        self.idxs[largest_class] = list(range(self.classes_items[largest_class]))

        for i in range(self.classes):
            if i == largest_class:
                continue

            part_required_size = self.unique_batches * self.quants[i]

            copies = (part_required_size + self.classes_items[i] - 1)//self.classes_items[i]
            self.idxs[i] = (list(range(self.classes_items[i]))*copies)[:part_required_size]

    def __iter__(self) -> Iterator[List[int]]:
        self.prepare_idxs()
        for i in range(len(self.idxs)):
            shuffle(self.idxs[i])
        for b in range(self.unique_batches):
            batch = []
            for p in range(self.classes):
                for _ in range(self.quants[p]):
                    if len(self.idxs[p]) > 0:
                        batch.append((p, self.idxs[p].pop()))
            yield batch

    def __len__(self) -> int:
        self.classes_items = self.dataset.labels_stats()
        f = lambda i: self.classes_items[i]
        largest_class = max(range(self.classes), key=f)

        if self.drop_last:
            return self.classes_items[largest_class] // self.quants[largest_class]
        else:
            return (self.classes_items[largest_class] +
                    self.quants[largest_class] - 1) // self.quants[largest_class]


class BalancedDataLoader(DataLoader):

    def __init__(self, dataset: PerClassFilesDataset, classes: int = 2, batch_size: Optional[int] = 1,
                 shuffle: bool = False, sampler: Union[Sampler, Iterable, None] = None,
                 batch_sampler: Union[Sampler[Sequence], Iterable[Sequence], None] = None,
                 num_workers: int = 0, collate_fn: Optional[_collate_fn_t] = None,
                 pin_memory: bool = False, drop_last: bool = False,
                 timeout: float = 0, worker_init_fn: Optional[_worker_init_fn_t] = None,
                 multiprocessing_context=None, generator=None,
                 *, prefetch_factor: int = 2,
                 persistent_workers: bool = False):
        super(BalancedDataLoader, self).__init__(dataset=dataset,
                                                  batch_sampler=BalancedBatchSampler(dataset=dataset,
                                                                                     classes=classes,
                                                                                     batch_size=batch_size,
                                                                                     drop_last=drop_last),
                                                  num_workers=num_workers, collate_fn=collate_fn,
                                                  pin_memory=pin_memory,
                                                  timeout=timeout, worker_init_fn=worker_init_fn,
                                                  multiprocessing_context=multiprocessing_context, generator=generator,
                                                  prefetch_factor=prefetch_factor,
                                                  persistent_workers=persistent_workers)


    def total_batches(self):
        return len(self.batch_sampler)   # type: ignore[arg-type]



