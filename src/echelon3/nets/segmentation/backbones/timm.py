from typing import List
import torch.nn as nn
import torch
import timm


class TimmSegmentationBackbone(nn.Module):

    def __init__(self, model, pretrained=True, in_chans: int = 3, features_layers=(4, 7, 14, 18)):
        super(TimmSegmentationBackbone, self).__init__()

        self.net = timm.create_model(features_only=True, in_chans=in_chans,
                                     model_name=model, pretrained=pretrained, out_indices=features_layers)

#        from torchvision.models import mobilenet_v2

#        mobilenet = mobilenet_v2(weights=weights)

 #       self.feature_layers = nn.ModuleList()
#        last = 0
#        for l in features_layers:
#            self.feature_layers.append(mobilenet.features[last:l])
#            last = l

    def forward(self, x):
        xs = self.net(x)
#        xs = []
#        last = x
#        for l in self.feature_layers:
#            last = l(last)
#            xs.append(last)
#
        return xs
