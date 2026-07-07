"""Tiny FCN backbone for the segmentation smoke.

`Segmenter` runs `backbone(x)` -> `neck` -> `head`, then bilinearly upsamples the
head output back to the input resolution. This backbone returns a single feature
map with `channels` channels at 1/4 resolution; pair it with
`SimpleHead(channels, num_classes)`. Any backbone producing a single feature
tensor plugs in the same way (a real run would use a timm/DDRNet/TopFormer
backbone). The CLI puts the current directory on `sys.path`, so running from
`examples/segmentation/` makes `module: backbone` resolve to this file.
"""
import torch.nn as nn


def _cbr(cin, cout, stride=1):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class TinySegBackbone(nn.Module):
    def __init__(self, channels=32, **kwargs):
        super().__init__()
        self.body = nn.Sequential(
            _cbr(3, channels, stride=2),        # /2
            _cbr(channels, channels, stride=2), # /4
            _cbr(channels, channels),
            _cbr(channels, channels),
        )

    def forward(self, x):
        return self.body(x)
