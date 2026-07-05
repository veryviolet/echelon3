import torch
import torch.nn as nn
from torch import Tensor


class WMV_Ur_Loss(nn.Module):

    def __init__(self, p, gamma, alpha=None, beta=1.0):
        super(WMV_Ur_Loss, self).__init__()
        self.p = p
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
        if len(inputs.shape) > 1:
            if inputs.shape[1] == 2:
                scores = torch.softmax(inputs, dim=-1)[:, 1]
            else:
                scores = torch.squeeze(inputs)
        else:
            scores = inputs

        if self.alpha is not None:
            mean_score = torch.mean(scores)
            scaled_scores = torch.zeros_like(scores)
            scaled_scores[scores > self.beta*mean_score] = \
                torch.pow(scores[scores > self.beta*mean_score]-self.beta*mean_score, self.alpha)
            scores = scaled_scores

        positive = scores[target == 1]
        negative = scores[target == 0]
        positive_rep = positive.unsqueeze(1).repeat(1, negative.size(0))
        negative_rep = negative.unsqueeze(1).repeat(1, positive.size(0)).T
        diff = positive_rep - negative_rep
        selected_diff = diff[diff < self.gamma]
        rs = torch.pow(-(selected_diff-self.gamma), self.p)
        return torch.sum(rs)/inputs.size(0)



class WeightedCrossEntropyLoss(nn.Module):

    def __init__(self, weight):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.base_loss = nn.CrossEntropyLoss(weight=Tensor(tuple(weight)))

    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
        if self.base_loss.weight.device != inputs.device:
            self.base_loss.weight = self.base_loss.weight.to(inputs.device)
        return self.base_loss(inputs, target)


class FixedMarginCrossEntropyLoss(nn.Module):
    
    def __init__(self, m: float, idx: int):
        super(FixedMarginCrossEntropyLoss, self).__init__()

        self.m = m
        self.idx = idx
        self.base_loss = nn.CrossEntropyLoss()

    def forward(self, inputs: Tensor, target: Tensor) -> Tensor:
        mods = inputs + 1e-5
        mods[target == self.idx, self.idx] = mods[target == self.idx, self.idx] - self.m
 #       mods = (target == self.idx).float() * self.m
#        res = input - mods

        return self.base_loss(mods, target)

