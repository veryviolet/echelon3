"""A tiny fully-convolutional heatmap backbone for the detector smoke.

`HeatmapDetector` needs a backbone that maps an image to `object_classes + 2`
channels at `output_size` (the class heatmaps plus width/height). Any such module
plugs in the same way — this toy keeps the example dependency-free (no mmcv /
TopFormer) and doubles as a demonstration of referencing your own network from a
config by import path: the CLI puts the current directory on `sys.path`, so
running from `examples/detector/` makes `module: backbone` resolve.
"""
import torch.nn as nn
import torch.nn.functional as F


def _cbr(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class TinyHeatmapBackbone(nn.Module):
    def __init__(self, out_channels, width=32, output_size=(128, 128), **kwargs):
        super().__init__()
        self.output_size = tuple(output_size)
        self.body = nn.Sequential(
            _cbr(3, width, stride=2),          # /2
            _cbr(width, width * 2, stride=2),  # /4
            _cbr(width * 2, width * 2),
            _cbr(width * 2, width * 2),
        )
        self.head = nn.Conv2d(width * 2, out_channels, 1)

    def forward(self, x):
        f = self.body(x)
        f = self.head(f)
        return F.interpolate(f, size=self.output_size, mode='bilinear', align_corners=False)
