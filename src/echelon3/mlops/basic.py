from abc import abstractmethod
from typing import Dict, List, Union
import torch


class MLOpsLogger:

    @abstractmethod
    def log_train_data(self, step, source: torch.Tensor, labels: torch.Tensor, predictions: torch.Tensor):
        pass

    @abstractmethod
    def log_test_data(self, step, source: torch.Tensor, labels: torch.Tensor, predictions: torch.Tensor):
        pass

    @abstractmethod
    def log_train_losses(self, step, losses: Dict[str, torch.Tensor]):
        pass

    @abstractmethod
    def log_test_losses(self, step, losses: Dict[str, torch.Tensor]):
        pass

    @abstractmethod
    def log_train_metrics(self, step, metrics: Dict[str, torch.Tensor]):
        pass

    @abstractmethod
    def log_test_metrics(self, step, metrics: Dict[str, torch.Tensor]):
        pass

    @abstractmethod
    def start(self):
        pass

    @abstractmethod
    def finalize(self):
        pass
