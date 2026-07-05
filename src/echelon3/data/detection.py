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


class DetectionDataset(AllFilesDataset):

    classes = None
    image_extension = None
    annotations_extension = None
    image_folder = None
    annotations_folder = None

    bboxes_type = None
    process_anno_fn = None

    @staticmethod
    def process_on_anno_yolo(annostr):
        cls, xc, yc, width, height = str(annostr).split(' ')
        cls = int(cls)
        xc = float(xc)
        yc = float(yc)
        width = float(width)
        height = float(height)
        return xc, yc, width, height, cls

    @staticmethod
    def process_on_anno_coco(annostr):
        pars = [int(x) for x in str(annostr).split(' ')]
        return pars[1], pars[2], pars[3], pars[4], pars[0]

    @staticmethod
    def process_on_anno_pascal_voc(annostr):
        pars = [int(x) for x in str(annostr).split(' ')]
        return pars[1], pars[2], pars[3], pars[4], pars[0]

    @staticmethod
    def process_on_anno_albumentations(annostr):
        cls, xmin, ymin, xmax, ymax = str(annostr).split(' ')
        cls = int(cls)
        xmin = float(xmin)
        ymin = float(ymin)
        xmax = float(xmax)
        ymax = float(ymax)
        return xmin, ymin, xmax, ymax, cls

    def __init__(self, augment, preprocess, label_type='bboxes', bboxes_type='yolo', colors=None, **kwargs):
        super(DetectionDataset, self).__init__(augment=augment, preprocess=preprocess, label_type=label_type,
                                                        **kwargs)
        self.class_colors = None
        self.bboxes_type = bboxes_type

        if bboxes_type == 'yolo':
            self.process_anno_fn = DetectionDataset.process_on_anno_yolo
        elif bboxes_type == 'coco':
            self.process_anno_fn = DetectionDataset.process_on_anno_coco
        elif bboxes_type == 'pascal_voc':
            self.process_anno_fn = DetectionDataset.process_on_anno_pascal_voc
        elif bboxes_type == 'albumentations':
            self.process_anno_fn = DetectionDataset.process_on_anno_albumentations
        else:
            raise RuntimeError(f'unknown bboxes type: {bboxes_type}')

    def process_label(self, label, image):
        annotations = []

        with open(label, 'r') as f:
            for one in f:
                try:
                    anno = self.process_anno_fn(one)
                    annotations.append(anno)
                except Exception as e:
                    continue

        return annotations

    def collect_filenames_with_labels(self):

        total = []

        for f in glob.glob(os.path.join(self.image_folder, '**', '*.' + self.image_extension), recursive=True):
            m = f.replace(self.image_folder, self.annotations_folder).\
                replace('.'+self.image_extension, '.'+self.annotations_extension)
            if not os.path.exists(m):
                m = ''
            total.append((f, m))

        shuffle(total)

        self.filenames_with_labels = total

