from abc import abstractmethod
import torch


class Metric:

    def to(self, *args, **kwargs):
        # Custom metrics accumulate on CPU (numpy/scipy) or move tensors to the
        # right device inside update(); the trainer calls metric.to(device)
        # uniformly (torchmetrics need it), so this keeps them interchangeable.
        #
        # НО: если метрика — ещё и nn.Module с параметрами/буферами, этот no-op
        # затирал бы nn.Module.to по MRO, буферы оставались бы на CPU и validate на
        # cuda падал бы (RuntimeError device mismatch). Поэтому делегируем в
        # nn.Module.to, когда self — модуль.
        if isinstance(self, torch.nn.Module):
            return torch.nn.Module.to(self, *args, **kwargs)
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
