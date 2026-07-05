import torch
import torch
import torch.nn.functional as F

class BasicWrapper(torch.nn.Module):

    def __init__(self, core, **kwargs):
        super(BasicWrapper, self).__init__()

        self.core = core

