from typing import Optional, Union

import albumentations
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.io as io
import torchvision.transforms
from torch import device
from torch.nn.modules.module import T


class ToTensor(torch.nn.Module):

    def __init__(self, **kwargs):
        super(ToTensor, self).__init__()

    def forward(self, x):
        x = torch.from_numpy(x)
        x = x.permute(2, 0, 1)
        return x



class Id(torch.nn.Module):

    def __init__(self, **kwargs):
        super(Id, self).__init__()

    def forward(self, x):
        return x


class To01(torch.nn.Module):

    def __init__(self, **kwargs):
        super(To01, self).__init__()


    def forward(self, x):
        x = x.float()
        x = x / 255.0
        return x


class Decode(torch.nn.Module):

    def __init__(self, **kwargs):
        super(Decode, self).__init__()

    def forward(self, x):
        return io.decode_image(x)


class Squeeze(torch.nn.Module):

    def __init__(self, dim=0, **kwargs):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        x = torch.squeeze(x, dim=self.dim)
        return x


class ToCHW(torch.nn.Module):

    def __init__(self, **kwargs):
        super(ToCHW, self).__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class ToHWC(torch.nn.Module):

    def __init__(self, **kwargs):
        super(ToHWC, self).__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)



class PatchesToCHW(torch.nn.Module):

    def __init__(self, **kwargs):
        super(PatchesToCHW, self).__init__()

    def forward(self, x):
        return x.permute(0, 1, 4, 2, 3)




class SmallestMaxSize(torch.nn.Module):

    def __init__(self, max_size, interpolation=0, **kwargs):
        super(SmallestMaxSize, self).__init__()

        self.max_size = max_size
        self.interpolation = interpolation

    def forward(self, x):
        if x.size(-1) >= x.size(-2):
            result_size = [self.max_size, (x.size(-1)*self.max_size)//x.size(-2)]
        else:
            result_size = [(x.size(-2) * self.max_size) // x.size(-1), self.max_size]
        return torchvision.transforms.functional.resize(x, result_size)


class Resize(torch.nn.Module):

    def __init__(self, size, interpolation='nearest', **kwargs):
        super(Resize, self).__init__()

        self.size = size
        self.interpolation = interpolation

    def forward(self, x):
        return torchvision.transforms.functional.resize(x, size=tuple(self.size),
                            interpolation=torchvision.transforms.functional.InterpolationMode(self.interpolation))




class ScoreFromLogits(torch.nn.Module):

    def __init__(self, score_class, **kwargs):
        super(ScoreFromLogits, self).__init__()

        self.score_class = score_class
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.softmax(x)
        return x[:, self.score_class]


class Normalize(torch.nn.Module):

    def __init__(self, mean, std, max_pixel, **kwargs):
        super(Normalize, self).__init__()

        self._mean = torch.tensor(mean) * max_pixel
        self._std = torch.tensor(std) * max_pixel

        if self._mean.ndim == 1:
            self._mean = self._mean.view(-1, 1, 1)
        if self._std.ndim == 1:
            self._std = self._std.view(-1, 1, 1)

        self.denominator = torch.reciprocal(self._std)

    def _apply(self, fn, **kwargs):
        super(Normalize, self)._apply(fn, **kwargs)
        self._mean = fn(self._mean)
        self._std = fn(self._std)
        return self
    def forward(self, x):
        x = x.float()
        x = x.sub(self._mean)
        x = x.div(self._std)
        return x


class CropToAspectRatio(torch.nn.Module):

    def __init__(self, height, width, **kwargs):
        super(CropToAspectRatio, self).__init__()
        self.height = height
        self.width = width
        self.aspect_ratio = (1.0*self.height)/self.width

    def forward(self, x):
        source_height = x.size(1)
        source_width = x.size(2)

        greater = (1.0*source_height)/source_width >= self.aspect_ratio

        if greater:
            new_width = source_width
            new_height = (new_width * self.height)//self.width
            delta = np.abs((source_height - new_height))//2
            res = x[:, delta:(delta+new_height), :]
            return res
        else:
            new_height = source_height
            new_width = (new_height * self.width)//self.height
            delta = np.abs((source_width - new_width))//2
            res = x[:, :, delta:(delta+new_width)]
            return res


class ToGrayscale(torch.nn.Module):
    def __init__(self):
        super(ToGrayscale, self).__init__()
        # Coefficients used for RGB to Grayscale conversion
        # These are based on ITU-R BT.601 standard: 0.2989 * R + 0.5870 * G + 0.1140 * B
        self.weights = torch.tensor([0.2989, 0.5870, 0.1140], dtype=torch.float32).view(1, 3, 1, 1)

    def forward(self, x):
        """
        Converts an RGB image batch to grayscale.
        :param x: Input image tensor of shape (B, 3, H, W)
        :return: Grayscale image tensor of shape (B, 1, H, W)
        """
        if x.shape[1] != 3:
            raise ValueError("Input tensor must have 3 channels (RGB).")

        # Move weights to the same device as input
        weights = self.weights.to(x.device)

        # Apply weights and sum along the channel dimension to produce grayscale
        grayscale = torch.nn.functional.conv2d(x, weights, bias=None, stride=1, padding=0, groups=x.size(0))

        grayscale_3_channels = grayscale.repeat(1, 3, 1, 1)

        return grayscale_3_channels

class GrayscaleCLAHE(torch.nn.Module):
    """Provider-invariant appearance: CHW RGB -> grayscale (+optional CLAHE) -> 3ch.

    Applied symmetrically to base & query via transform.preprocess (the dataset runs
    preprocess on both inputs), deterministically. I/O: CHW float in [0, 255] — place
    BEFORE Normalize. Cross-provider differences are mostly colour/brightness; this makes
    the input provider-invariant. Mirror it at inference time (raven.py).
    """

    def __init__(self, clahe: bool = True, clip: float = 2.0, grid: int = 8, **kwargs):
        super(GrayscaleCLAHE, self).__init__()
        self._clahe = (
            cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(grid), int(grid)))
            if clahe
            else None
        )

    @torch.no_grad()
    def forward(self, x):
        # x: [3, H, W] float in [0, 255]
        rgb = np.transpose(x.detach().cpu().numpy(), (1, 2, 0))
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if self._clahe is not None:
            gray = self._clahe.apply(gray)
        out = np.repeat(gray[None, :, :], 3, axis=0).astype(np.float32)
        return torch.from_numpy(out).to(device=x.device, dtype=x.dtype)
