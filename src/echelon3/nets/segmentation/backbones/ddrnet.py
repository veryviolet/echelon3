"""DDRNet-23-slim backbone, ported from
https://github.com/ydhongHIT/DDRNet (Pan et al. 2021).

Original was a full segmenter (backbone + DAPPM + seg-head). For our echelon3
multi-binary-head pipeline we keep the backbone + DAPPM fusion and return the
fused high-resolution feature map (1/8 of input, 64 channels for the slim
variant). The MultiBinarySegmenter then runs its per-head 1x1 conv on this
and bilinear-upsamples to input resolution.

INT8-quantization friendly: only conv/BN/ReLU/AvgPool/AdaptiveAvgPool/bilinear-upsample
operators. ~5.7M params for the slim variant (planes=32).

`forward` returns a list of length 1 (the fused feature) so it stays
compatible with `MultiBinarySegmenter`'s `feature_index=-1` selection.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

# DataParallel-compatible BN. Swap to SyncBatchNorm for DistributedDataParallel.
BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.no_relu = no_relu

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        return out if self.no_relu else self.relu(out)


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.no_relu = no_relu

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        return out if self.no_relu else self.relu(out)


class DAPPM(nn.Module):
    """Deep Aggregation Pyramid Pooling Module."""
    def __init__(self, inplanes, branch_planes, outplanes):
        super().__init__()
        def _scale_block(pool):
            return nn.Sequential(
                pool,
                BatchNorm2d(inplanes, momentum=bn_mom),
                nn.ReLU(inplace=True),
                nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
            )
        self.scale1 = _scale_block(nn.AvgPool2d(5, 2, 2))
        self.scale2 = _scale_block(nn.AvgPool2d(9, 4, 4))
        self.scale3 = _scale_block(nn.AvgPool2d(17, 8, 8))
        self.scale4 = _scale_block(nn.AdaptiveAvgPool2d((1, 1)))
        self.scale0 = nn.Sequential(
            BatchNorm2d(inplanes, momentum=bn_mom), nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )
        def _process():
            return nn.Sequential(
                BatchNorm2d(branch_planes, momentum=bn_mom), nn.ReLU(inplace=True),
                nn.Conv2d(branch_planes, branch_planes, kernel_size=3, padding=1, bias=False),
            )
        self.process1 = _process()
        self.process2 = _process()
        self.process3 = _process()
        self.process4 = _process()
        self.compression = nn.Sequential(
            BatchNorm2d(branch_planes * 5, momentum=bn_mom), nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, kernel_size=1, bias=False),
        )
        self.shortcut = nn.Sequential(
            BatchNorm2d(inplanes, momentum=bn_mom), nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        xs = [self.scale0(x)]
        xs.append(self.process1(F.interpolate(self.scale1(x), size=(h, w), mode="bilinear", align_corners=False) + xs[0]))
        xs.append(self.process2(F.interpolate(self.scale2(x), size=(h, w), mode="bilinear", align_corners=False) + xs[1]))
        xs.append(self.process3(F.interpolate(self.scale3(x), size=(h, w), mode="bilinear", align_corners=False) + xs[2]))
        xs.append(self.process4(F.interpolate(self.scale4(x), size=(h, w), mode="bilinear", align_corners=False) + xs[3]))
        return self.compression(torch.cat(xs, dim=1)) + self.shortcut(x)


class DDRNet23SlimBackbone(nn.Module):
    """DDRNet-23-slim backbone. Returns list with 1 tensor at /8 input resolution,
    with `out_channels` channels (default 64 for the slim variant)."""

    def __init__(self, planes: int = 32, spp_planes: int = 128,
                 pretrained_path: str | None = None, **kwargs):
        super().__init__()
        self.planes = planes
        highres_planes = planes * 2
        block = BasicBlock

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, planes, 3, stride=2, padding=1, bias=False),
            BatchNorm2d(planes, momentum=bn_mom), nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, 3, stride=2, padding=1, bias=False),
            BatchNorm2d(planes, momentum=bn_mom), nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU(inplace=False)

        self.layer1 = self._make_layer(block, planes, planes, 2)
        self.layer2 = self._make_layer(block, planes, planes * 2, 2, stride=2)
        self.layer3 = self._make_layer(block, planes * 2, planes * 4, 2, stride=2)
        self.layer4 = self._make_layer(block, planes * 4, planes * 8, 2, stride=2)

        self.compression3 = nn.Sequential(
            nn.Conv2d(planes * 4, highres_planes, 1, bias=False),
            BatchNorm2d(highres_planes, momentum=bn_mom),
        )
        self.compression4 = nn.Sequential(
            nn.Conv2d(planes * 8, highres_planes, 1, bias=False),
            BatchNorm2d(highres_planes, momentum=bn_mom),
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(highres_planes, planes * 4, 3, stride=2, padding=1, bias=False),
            BatchNorm2d(planes * 4, momentum=bn_mom),
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(highres_planes, planes * 4, 3, stride=2, padding=1, bias=False),
            BatchNorm2d(planes * 4, momentum=bn_mom), nn.ReLU(inplace=True),
            nn.Conv2d(planes * 4, planes * 8, 3, stride=2, padding=1, bias=False),
            BatchNorm2d(planes * 8, momentum=bn_mom),
        )
        self.layer3_ = self._make_layer(block, planes * 2, highres_planes, 2)
        self.layer4_ = self._make_layer(block, highres_planes, highres_planes, 2)
        self.layer5_ = self._make_layer(Bottleneck, highres_planes, highres_planes, 1)
        self.layer5 = self._make_layer(Bottleneck, planes * 8, planes * 8, 1, stride=2)
        self.spp = DAPPM(planes * 16, spp_planes, planes * 4)

        # public out-channels for downstream config awareness
        self.out_channels = planes * 4   # 128 for planes=32

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        if pretrained_path:
            self._load_pretrained(pretrained_path)

    def _load_pretrained(self, path: str) -> None:
        """Load ImageNet pretrained weights (e.g. DDRNet23s_imagenet.pth).
        Tolerates: 'module.' prefix from DataParallel, missing keys (classifier head),
        unexpected keys (linear/down5/last_layer), and shape mismatches (we use
        bias=False where pretrain used bias=True — bias is just skipped)."""
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        sd = {k.removeprefix("module."): v for k, v in sd.items()}
        own = self.state_dict()
        filtered = {k: v for k, v in sd.items()
                    if k in own and tuple(own[k].shape) == tuple(v.shape)}
        missing = sorted(set(own.keys()) - set(filtered.keys()))
        unexpected = sorted(set(sd.keys()) - set(own.keys()))
        skipped_shape = sorted(
            k for k in sd if k in own and tuple(own[k].shape) != tuple(v.shape for v in [sd[k]])[0]
        )
        self.load_state_dict(filtered, strict=False)
        print(f"[DDRNet] pretrained: loaded {len(filtered)}/{len(own)} params; "
              f"missing={len(missing)}, unexpected={len(unexpected)}")

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )
        layers = [block(inplanes, planes, stride, downsample)]
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            no_relu = (i == blocks - 1)
            layers.append(block(inplanes, planes, stride=1, no_relu=no_relu))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        h_out = x.shape[-2] // 8
        w_out = x.shape[-1] // 8

        x = self.conv1(x)
        l1 = self.layer1(x)
        l2 = self.layer2(self.relu(l1))
        l3 = self.layer3(self.relu(l2))
        x_ = self.layer3_(self.relu(l2))

        x = l3 + self.down3(self.relu(x_))
        x_ = x_ + F.interpolate(
            self.compression3(self.relu(l3)), size=(h_out, w_out),
            mode="bilinear", align_corners=False,
        )

        l4 = self.layer4(self.relu(x))
        x_ = self.layer4_(self.relu(x_))

        x = l4 + self.down4(self.relu(x_))
        x_ = x_ + F.interpolate(
            self.compression4(self.relu(l4)), size=(h_out, w_out),
            mode="bilinear", align_corners=False,
        )

        x_ = self.layer5_(self.relu(x_))
        x = F.interpolate(
            self.spp(self.layer5(self.relu(x))),
            size=(h_out, w_out), mode="bilinear", align_corners=False,
        )
        fused = x + x_       # (B, planes*4, H/8, W/8) — 128 channels for slim
        return [fused]
