import torch
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from scipy.special import softmax
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from torchmetrics.detection import mean_ap

from torch import Tensor
import torch.nn as nn

from echelon3.metrics.base import Metric
from echelon3.utils.bboxes import BBoxes, BBoxType, coords_to_absolute

BBOX_TYPE_TO_FORMAT = {
    BBoxType.YOLO: 'cxcywh',
    BBoxType.COCO: 'xywh',
    BBoxType.PASCAL_VOC: 'xyxy',
    BBoxType.ALBUMENTATIONS: 'xyxy'
}


class mAP(Metric):

    engine = None

    def __init__(self, bbox_type='yolo'):
        self.bbox_type = BBoxType(bbox_type)
        self.engine = mean_ap.MeanAveragePrecision(box_format=BBOX_TYPE_TO_FORMAT[self.bbox_type])
        self.reset()

    def update(self, predicted: torch.Tensor, target: torch.Tensor):
        output_size = predicted['heatmaps'].shape[2:]
        device = predicted['heatmaps'].device
        addon_predicted_list = [{'boxes': coords_to_absolute(predicted['bboxes'][idx, :, :], output_size).to(device),
                                 'scores': torch.squeeze(predicted['scores'][idx, :, :], dim=-1).to(device),
                                 'labels': torch.squeeze(predicted['classes'][idx, :, :], dim=-1).to(device)}
                                for idx in range(len(target))]

        addon_target_list = [{'boxes': coords_to_absolute(torch.Tensor([l[0:4] for l in target[idx]]), output_size).to(device),
                              'labels': torch.Tensor([l[4] for l in target[idx]]).to(device)} for idx in range(len(target))]

        self.engine.update(addon_predicted_list, addon_target_list)


    def compute(self):
        return self.engine.compute()['map']

    def reset(self):
        self.engine.reset()


