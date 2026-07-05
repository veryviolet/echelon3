import torch
import torch.nn as nn
import timm
import numpy as np

class CoreTransformer(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True):
        super().__init__()

        self.model = timm.create_model(model_name, pretrained=pretrained)

        self.model.patch_embed = nn.Identity()


    def forward(self, patches: torch.Tensor):
        return self.model.blocks(patches)

