from abc import abstractmethod
import torch


class Metric:

    def to(self, *args, **kwargs):
        # Custom metrics accumulate on CPU (numpy/scipy) or move tensors to the
        # right device inside update(); the trainer calls metric.to(device)
        # uniformly (torchmetrics need it), so this keeps them interchangeable.
        return self

    @abstractmethod
    def update(self, predicted: torch.Tensor, target: torch.Tensor):
        pass

    @abstractmethod
    def compute(self):
        pass

    @abstractmethod
    def reset(self):
        pass
