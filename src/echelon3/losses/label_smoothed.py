import torch.nn as nn
from torch import Tensor


class LabelSmoothedWeightedCrossEntropyLoss(nn.Module):
    """Drop-in extension of :class:`WeightedCrossEntropyLoss` that exposes
    PyTorch's built-in ``label_smoothing`` parameter from
    :class:`torch.nn.CrossEntropyLoss` (available since PyTorch 1.10).

    Use when an over-confident classifier on imbalanced multi-class data
    starts to overfit, i.e. train loss keeps decreasing while validation
    macro F1 plateaus or regresses. Setting ``label_smoothing=0.0`` makes
    this class behaviourally identical to ``WeightedCrossEntropyLoss``.
    """

    def __init__(self, weight, label_smoothing: float = 0.0):
        super(LabelSmoothedWeightedCrossEntropyLoss, self).__init__()
        self.base_loss = nn.CrossEntropyLoss(
            weight=Tensor(tuple(weight)),
            label_smoothing=float(label_smoothing),
        )

    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
        if self.base_loss.weight.device != inputs.device:
            self.base_loss.weight = self.base_loss.weight.to(inputs.device)
        return self.base_loss(inputs, target)
