from omegaconf import DictConfig
from echelon3.creator import create_universal
import torch
import torch.nn.functional as F


class Segmenter(torch.nn.Module):

    def __init__(self,
                 backbone: DictConfig,
                 head: DictConfig,
                 neck: DictConfig = None,
                 align_corners: bool = True,
                 inference: bool = False,
                 **kwargs):
        super(Segmenter, self).__init__()

        self.backbone = create_universal(backbone)
        self.head = create_universal(head)
        self.neck = create_universal(neck) if neck is not None else torch.nn.Identity()
        self.align_corners = align_corners
        self.inference = inference

#    def to(self, *args, **kwargs):
#        super(Segmenter, self).to(*args, **kwargs)
#        self.backbone.to(*args, **kwargs)
#        self.head.to(*args, **kwargs)
#        self.neck.to(*args, **kwargs)
    def forward(self, x):
        features = self.backbone(x)
        pre_output = self.neck(features)
        raw_output = self.head(pre_output)

#        if self.training:
#            return raw_output
#        else:

        rescaled_out = F.interpolate(input=raw_output, size=x.shape[2:],
                                     mode='bilinear', align_corners=self.align_corners)

#        if self.inference:
#            rescaled_out = torch.argmax(rescaled_out, dim=1, keepdim=True).to(torch.uint8)

        return rescaled_out


class LightSegmenter(torch.nn.Module):

    def __init__(self,
                 backbone: DictConfig,
                 head: DictConfig,
                 align_corners: bool,
                 inference: bool = False,
                 **kwargs):
        super(Segmenter, self).__init__()

        self.backbone = create_universal(backbone)
        self.head = create_universal(head)
        self.align_corners = align_corners
        self.inference = inference

    def forward(self, x):
        features = self.backbone(x)
        raw_output = self.head(features)

        #        if self.training:
        #            return raw_output
        #        else:

        rescaled_out = F.interpolate(input=raw_output, size=x.shape[2:],
                                     mode='bilinear', align_corners=self.align_corners)

        if self.inference:
            rescaled_out = torch.argmax(rescaled_out, dim=1, keepdim=True).to(torch.uint8)

        return rescaled_out

