import os.path
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union, List
from collections import Counter
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from omegaconf import OmegaConf, DictConfig, ListConfig
import cv2
import numpy as np

from echelon3.creator import create_universal, create_single_dataset
from echelon3.data.filters import BaseFilter

FILTER_KEY = 'filter'

class BasicDataset(Dataset):

    augment = None
    preprocess = None

    label_type = None
    bboxes_type = None

    params_string = None

    def __init__(self, augment, preprocess, label_type='class', bboxes_type='yolo', **kwargs):
        super(BasicDataset, self).__init__()

        self.augment = augment
        self.preprocess = preprocess
        self.label_type = label_type
        self.bboxes_type = bboxes_type

        for key, val in kwargs.items():
            setattr(self, key, val)

    def __str__(self):
        return f'{type(self).__name__}(length={self.__len__()})'

    def get_source_path(self, idx: int) -> str | None:
        return None


class FilesDataset(BasicDataset):

    filter: BaseFilter = None

    def __init__(self, augment, preprocess, label_type='class', **kwargs):
        super(FilesDataset, self).__init__(augment=augment, preprocess=preprocess, label_type=label_type, **kwargs)

        if FILTER_KEY in kwargs.keys():
            self.filter = create_universal(kwargs[FILTER_KEY])

        self.collect_filenames_with_labels()

    @abstractmethod
    def collect_filenames_with_labels(self):
        pass

    @abstractmethod
    def get_item(self, idx):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def labels_stats(self):
        pass

    def process_label(self, label, image):
        label = cv2.imread(label) if os.path.exists(label) else np.zeros_like(image)
        return label[:, :, 0].astype(np.int64)

    def __getitem__(self, idx):

        data = None

        try:
            data = self.get_item(idx)

            image = cv2.imread(data[0])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if self.label_type != 'class':
                label = self.process_label(data[1], data[0])
            else:
                label = data[1]

            if self.augment is not None:
                if self.label_type == 'class':
                    transformed = self.augment(image=image)
                    transformed_label = label
                elif self.label_type == 'image':
                    transformed = self.augment(image=image, mask=label)
                    transformed_label = transformed['mask'].long()
                elif self.label_type == 'bboxes':
                    transformed = self.augment(image=image, bboxes=label)
                    transformed_label = transformed['bboxes']
                else:
                    raise RuntimeError(f'unknown label type {self.label_type}')

                transformed_image = transformed['image']

            else:
                transformed_image = image
                transformed_label = label

            preprocessed_image = self.preprocess(transformed_image) \
                if self.preprocess is not None else transformed_image

        except Exception as e:
            raise RuntimeError(f'failed to process {data} with exception {e}')

        return preprocessed_image, transformed_label


class AllFilesDataset(FilesDataset):
    filenames_with_labels: List[Tuple[str, Union[int, str]]] = None

    def __len__(self):
        return len(self.filenames_with_labels)

    def labels_stats(self):
        return dict(Counter([s[1] for s in self.filenames_with_labels]))

    def get_item(self, idx):
        return self.filenames_with_labels[idx]

    @abstractmethod
    def collect_filenames_with_labels(self):
        pass


class PerClassFilesDataset(FilesDataset):
    filenames: Dict[Union[int, str], List[str]] = None

    def __len__(self):
        return sum([len(self.filenames[c]) for c in self.filenames.keys()])

    def labels_stats(self):
        return {c: len(self.filenames[c]) for c in self.filenames.keys()}

    def get_item(self, idx: Union[int, Tuple[int, int]]):
        if type(idx) == tuple:
            return self.filenames[idx[0]][idx[1]], idx[0]
        elif type(idx) == int:
            total = 0
            for lab in self.filenames.keys():
                total += len(self.filenames[lab])
                if idx < total:
                    break
            return self.filenames[lab][total-idx-1], lab
        else:
            raise RuntimeError('index should be either int or tuple of ints')

    @abstractmethod
    def collect_filenames_with_labels(self):
        pass


class MultiPartDataset:

    parts: List[Tuple[float, BasicDataset]] = None

    def __init__(self, parts: ListConfig, augment, preprocess, label_type='class', **kwargs):
        self.parts = []
        for p in parts:
            self.parts.append((p.share, create_single_dataset(p, augment=augment, preprocess=preprocess,
                                                                 label_type=label_type)))

    def parts_num(self):
        return len(self.parts)

    def part_share(self, part):
        return self.parts[part][0]

    def part_len(self, part):
        return len(self.parts[part][1])

    def max_part(self):
        f = lambda i: len(self.parts[i][1])
        return max(range(len(self.parts)), key=f)

    def __getitem__(self, idx: Tuple[int, int]):
        return self.parts[idx[0]][1][idx[1]]

    def __len__(self):
        return self.part_len(self.max_part())*self.parts_num()


class ClasswiseDataset:

    per_class_datasets: Dict[int, BasicDataset] = None

    def __init__(self, class_datasets: DictConfig, augment, preprocess, label_type='class', **kwargs):
        self.per_class_datasets = {}
        for label, ds in class_datasets.items():
            self.per_class_datasets[int(label)] = create_single_dataset(ds, augment=augment, preprocess=preprocess,
                                                                        label_type=label_type)

    def labels_stats(self):
        return {label: len(self.per_class_datasets[label]) for label in self.per_class_datasets.keys()}

    def classes_num(self):
        return len(self.per_class_datasets)

    def classes_keys(self):
        return list(self.per_class_datasets.keys())

    def class_ds_len(self, label):
        return len(self.per_class_datasets[label])

    def class_ds(self, label):
        return self.per_class_datasets[label]

    def __getitem__(self, idx: Tuple[int, int]):
        return self.per_class_datasets[idx[0]][idx[1]]

    def __len__(self):
        return sum([self.class_ds_len(label) for label in self.per_class_datasets.keys()])
    