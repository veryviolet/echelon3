from typing import Dict, Tuple
import torch
import os
import glob
from random import shuffle
import numpy as np
import cv2
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from omegaconf import OmegaConf, DictConfig

from echelon3.data.basic import AllFilesDataset


class ImageSegmentationDataset(AllFilesDataset):

    classes = None
    image_extension = None
    mask_extension = None
    image_folder = None
    mask_folder = None

    class_colors = None

    def __init__(self, augment, preprocess, label_type='image', class_colors=None, **kwargs):
        super(ImageSegmentationDataset, self).__init__(augment=augment, preprocess=preprocess, label_type=label_type,
                                                       **kwargs)
        self.class_colors = None
        if class_colors is not None:
            self.class_colors = {c: e for c, e in class_colors.items()}

    def process_label(self, label, image):
        label = cv2.imread(label) if os.path.exists(label) else np.zeros_like(image)

        if self.class_colors is None:
            return label[:, :, 0].astype(np.int64)
        else:
            res = label[:, :, 0].astype(np.int64)
            for c in self.class_colors.keys():
                res[(label == self.class_colors[c])[:, :, 0]] = c
            return res

    def collect_filenames_with_labels(self):

        total = []

        for f in glob.glob(os.path.join(self.image_folder, '**', '*.'+ self.image_extension), recursive=True):
            m = f.replace(self.image_folder, self.mask_folder).\
                replace('.'+self.image_extension, '.'+self.mask_extension)
            if not os.path.exists(m):
                m = ''
            total.append((f, m))

        shuffle(total)

        self.filenames_with_labels = total

    def get_source_path(self, idx: int) -> str | None:
        """Вернуть путь к исходному изображению по индексу датасета."""
        if hasattr(self, "filenames_with_labels") and 0 <= idx < len(self.filenames_with_labels):
            # кортеж вида (image_path, mask_path), возвращаем путь к изображению
            return self.filenames_with_labels[idx][0]
        return None