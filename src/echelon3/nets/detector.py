import torch.nn as nn
from omegaconf import DictConfig
from echelon3.creator import create_universal
import torch

from echelon3.utils.bbox_encode_decode import DecodeHeatmaps
from echelon3.utils.bboxes import BBoxType

class ModalityConv(nn.Module):

    def __init__(self, in_channels, bias=False):
        super(ModalityConv, self).__init__()
        self.k3 = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                   groups=in_channels, bias=bias, padding=1)
        self.k1 = nn.Conv2d(in_channels, in_channels,
                                   kernel_size=1, groups=in_channels, bias=bias)
        self.act = nn.Sigmoid()

    def forward(self, x):
        out = self.k3(x)
        out = self.k1(out)
        out = self.act(out)
        return out


class HeatmapDetector(torch.nn.Module):

    def __init__(self,
                 heatmap_backbone: DictConfig,
                 object_classes: int,
                 output_size: tuple = (224, 224),
                 **kwargs):
        super(HeatmapDetector, self).__init__()

        self.heatmap_backbone = create_universal(heatmap_backbone)

        self.object_classes = object_classes
        self.post_convs = ModalityConv(in_channels=object_classes+2)
#        self.pooler = nn.MaxPool2d(kernel_size=3)
        self.decoder = DecodeHeatmaps(num_classes=object_classes, output_size=output_size, bbox_type=BBoxType.YOLO)

    def forward(self, x):
        pre_heatmaps = self.heatmap_backbone(x)
        heatmaps = self.post_convs(pre_heatmaps)

        if not self.training:
            bboxes, scores, clses = self.decoder.decode(heatmaps[:, :-2, :, :], heatmaps[:, -2:, :, :])
            return {'heatmaps': heatmaps, 'bboxes': bboxes, 'scores': scores, 'classes': clses}

        return heatmaps

