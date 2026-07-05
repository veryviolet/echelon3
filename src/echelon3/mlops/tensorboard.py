from abc import abstractmethod
from typing import Dict, List, Union
import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from echelon3.utils.bboxes import BBoxes, BBoxType, BBoxesClass
# from echelon3.utils.bbox_encode_decode import EncodeBBoxes, DecodeHeatmaps

from echelon3.mlops.basic import MLOpsLogger

TRAIN_POSTFIX = 'train'
TEST_POSTFIX = 'test'


class TensorboardLogger(MLOpsLogger):

    def __init__(self, folder, **kwargs):
        self.train_summary_writer = SummaryWriter(os.path.join(folder, TRAIN_POSTFIX))
        self.test_summary_writer = SummaryWriter(os.path.join(folder, TEST_POSTFIX))

    def log_train_data(self, step, source: torch.Tensor, labels: torch.Tensor, predictions: torch.Tensor):
        pass

    def log_test_data(self, step, source: torch.Tensor, labels: torch.Tensor, predictions: torch.Tensor):
        pass

    def log_train_losses(self, step, losses: Dict[str, torch.Tensor]):
        for k, v in losses.items():
            self.train_summary_writer.add_scalar(k, v, global_step=step)

    def log_test_losses(self, step, losses: Dict[str, torch.Tensor]):
        for k, v in losses.items():
            self.test_summary_writer.add_scalar(k, v, global_step=step)

    def log_train_metrics(self, step, metrics: Dict[str, torch.Tensor]):
        for k, v in metrics.items():
            self.train_summary_writer.add_scalar(k, v, global_step=step)

    def log_test_metrics(self, step, metrics: Dict[str, torch.Tensor]):
        for k, v in metrics.items():
            self.test_summary_writer.add_scalar(k, v, global_step=step)

    def start(self):
        pass

    def finalize(self):
        pass


class ClassifierTensorboardLogger(TensorboardLogger):

    def log_train_data(self, step, source: torch.Tensor, labels: torch.Tensor, predictions: torch.Tensor):
        pass

    def log_test_data(self, step, source: torch.Tensor, labels: torch.Tensor, predictions: torch.Tensor):
        pass


class SegmentationTensorboardLogger(TensorboardLogger):

    def log_train_data(self, step, source: torch.Tensor, labels: torch.Tensor, predictions: torch.Tensor):
        pass

    def log_test_data(self, step, source: torch.Tensor, labels: torch.Tensor, predictions: torch.Tensor):
        pass


class DetectionTensorboardLogger(TensorboardLogger):

    def __init__(self, folder, num_classes=2, output_size=(224, 224),
                 source_bbox_type='yolo', predicted_bbox_type='yolo',
                 show_source_heatmaps=False,
                 class_to_show=0,
                 show_predicted_heatmaps=True, **kwargs):
        super(DetectionTensorboardLogger, self).__init__(folder=folder, **kwargs)

        self.num_classes = num_classes
        self.class_to_show = class_to_show
        self.output_size = output_size
        self.source_bbox_type = BBoxType(source_bbox_type)
        self.predicted_bbox_type = BBoxType(predicted_bbox_type) if predicted_bbox_type is not None else None
        self.show_source_heatmaps = show_source_heatmaps
        self.show_predicted_heatmaps = show_predicted_heatmaps

        # self.encoder = EncodeBBoxes(bbox_type=self.source_bbox_type, num_classes=num_classes,
        #                             output_size=self.output_size) if self.show_source_heatmaps else None
        # self.decoder = DecodeHeatmaps(bbox_type=self.predicted_bbox_type, num_classes=self.num_classes,
        #                               output_size=self.output_size) if self.show_predicted_heatmaps else None

    def log_train_data(self, step, source: torch.Tensor, labels: torch.Tensor, predictions: torch.Tensor):
        if self.show_source_heatmaps:
            all_heatmaps = self.encoder.generate(labels)

        for idx in range(source.shape[0]):
            source_boxes = BBoxesClass[self.source_bbox_type](labels[idx]).convert(
                BBoxType.PASCAL_VOC,
                size=source.shape[2:]
            )
            self.train_summary_writer.add_image_with_boxes(
                'bounding boxes',
                source[idx, :, :, :],
                source_boxes.data,
                global_step=step
            )

            if self.show_source_heatmaps:
                heatmap = 255 - cv2.normalize(
                    all_heatmaps[idx, self.class_to_show, :, :].numpy(),
                    None, alpha=0, beta=255,
                    norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
                )
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                self.train_summary_writer.add_image(
                    'heatmap',
                    torch.Tensor(heatmap).permute(2, 0, 1),
                    global_step=step
                )

    def log_test_data(self, step, source: torch.Tensor, labels: torch.Tensor, predictions: torch.Tensor):
        pass

