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
from echelon3.creator import create_universal

T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')
_worker_init_fn_t = Callable[[int], None]
_collate_fn_t = Callable[[List[T]], Any]


from echelon3.data.basic import PerClassFilesDataset, ClasswiseDataset
from echelon3.creator import create_batch_sampler


class ClasswiseBatchSampler(BatchSampler):

    batch_size = None
    drop_last = None
    classes = None
    idxs = None
    unique_batches = None
    dataset = None
    subbatches = None
    subsamplers = None

    def __init__(self, dataset: ClasswiseDataset, batch_size: int, drop_last: bool,
                 subsamplers) -> None:
        super().__init__(sampler=None, batch_size=batch_size, drop_last=drop_last)  # type: ignore[arg-type]

        self.dataset = dataset
        self.classes = dataset.classes_num()
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.quants = [self.batch_size // self.classes] * self.classes
        self.quants[list(self.dataset.classes_keys())[-1]] = self.batch_size - sum(self.quants[:-1])
        self.subsamplers = {l: create_batch_sampler(cfg, dataset=dataset.class_ds(l),
                                                    batch_size=self.quants[l], drop_last=drop_last)
                            for l, cfg in subsamplers.items()}
        pass

    def get_part_idx(self, part) -> int:
        for i in self.idxs[part]:
            yield i

    def prepare_idxs(self):
        for s in self.subsamplers.values():
            s.prepare_idxs()
        self.subbatches = {lab: v.unique_batches for lab, v in self.subsamplers.items()}
        f = lambda i: self.subbatches[i]
        largest_class = max(self.dataset.classes_keys(), key=f)

        self.unique_batches = self.subbatches[largest_class]

        self.idxs = {label: [] for label in self.dataset.classes_keys()}

        for batch in self.subsamplers[largest_class]:
            self.idxs[largest_class].append(batch)

        for label in self.dataset.classes_keys():
            if label == largest_class:
                continue

            part_required_size = self.unique_batches

            copies = (part_required_size + self.subbatches[label] - 1)//self.subbatches[label]

            prelist = []
            for batch in self.subsamplers[label]:
                prelist.append(batch)

            self.idxs[label] = (prelist*copies)[:part_required_size]

    def __iter__(self) -> Iterator[Dict[int, List[Tuple[int, int]]]]:
        self.prepare_idxs()
        for i in range(len(self.idxs)):
            shuffle(self.idxs[i])
        for b in range(self.unique_batches):
            batch = []
            for label in self.dataset.classes_keys():
                if len(self.idxs[label]) > 0:
                    prebatch = self.idxs[label].pop()
                    batch += [(label, p) for p in prebatch]
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


class ClasswiseDataLoader(DataLoader):

    per_class_dataloaders = None

    def __init__(self, dataset: ClasswiseDataset,
                 subsamplers: DictConfig,
                 classes: int = 2, batch_size: Optional[int] = 1,
                 shuffle: bool = False, sampler: Union[Sampler, Iterable, None] = None,
                 batch_sampler: Union[Sampler[Sequence], Iterable[Sequence], None] = None,
                 num_workers: int = 0, collate_fn: Optional[_collate_fn_t] = None,
                 pin_memory: bool = False, drop_last: bool = False,
                 timeout: float = 0, worker_init_fn: Optional[_worker_init_fn_t] = None,
                 multiprocessing_context=None, generator=None,
                 *, prefetch_factor: int = 2,
                 persistent_workers: bool = False):

        super(ClasswiseDataLoader, self).__init__(dataset=dataset,
                                                  batch_sampler=ClasswiseBatchSampler(dataset=dataset,
                                                                                      batch_size=batch_size,
                                                                                      drop_last=drop_last,
                                                                                      subsamplers=subsamplers),
                                                  num_workers=num_workers, collate_fn=collate_fn,
                                                  pin_memory=pin_memory,
                                                  timeout=timeout, worker_init_fn=worker_init_fn,
                                                  multiprocessing_context=multiprocessing_context, generator=generator,
                                                  prefetch_factor=prefetch_factor,
                                                  persistent_workers=persistent_workers)


    def total_batches(self):
        return len(self.batch_sampler)   # type: ignore[arg-type]



