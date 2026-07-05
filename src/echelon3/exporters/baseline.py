from abc import ABC, abstractmethod
import torch.nn as nn


class ModelExporter(ABC):

    preprocess: nn.Module = None
    postprocess: nn.Module = None
    target: str = None

    model_to_export: nn.Module = None

    def __init__(self, net: nn.Module, target: str,
                 preprocess: nn.Module, postprocess: nn.Module):
        class ModelToExport(nn.Module):

            def __init__(self, net, preprocess, postprocess):
                super(ModelToExport, self).__init__()

                self.basenet = net
                self.preprocess = preprocess
                self.postprocess = postprocess

            def forward(self, x):
                x = self.preprocess(x)
                x = self.basenet(x)
                x = self.postprocess(x)
                return x

        self.model_to_export = ModelToExport(net=net, preprocess=preprocess, postprocess=postprocess)
        self.target = target

    @abstractmethod
    def export(self):
        pass
