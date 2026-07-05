import torch
import torchvision.io as io
import torchvision.transforms


class ToSegmentationMask(torch.nn.Module):

    def __init__(self, **kwargs):
        super(ToSegmentationMask, self).__init__()

    def forward(self, x):
        x = torch.argmax(x, dim=1, keepdim=True).to(torch.uint8)
        return x


class ToBinarySegmentationMask(torch.nn.Module):

    def __init__(self, threshold = 0.5, **kwargs):
        super(ToBinarySegmentationMask, self).__init__()

        self.threshold = threshold

    def forward(self, x):
        x = torch.sigmoid(x)
        x = (x > self.threshold).to(torch.uint8)
        return x

