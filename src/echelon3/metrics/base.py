from abc import abstractmethod
import torch


class Metric:

    @abstractmethod
    def update(self, predicted: torch.Tensor, target: torch.Tensor):
        pass

    @abstractmethod
    def compute(self):
        pass

    @abstractmethod
    def reset(self):
        pass
