import torch
import torch.nn as nn
import torch.nn.functional as F
import omegaconf
from echelon3.nets.ops import resize
from echelon3.creator import create_universal

import numpy as np

class RescaledSegLoss(nn.Module):

    def __init__(self, wrapped_loss: omegaconf.dictconfig, **kwargs):
        super(RescaledSegLoss, self).__init__()
        self.wrapped_loss = create_universal(wrapped_loss)


    def forward(self, logits, labels):
        if logits.shape[2:] == labels.shape[1:]:
            return self.wrapped_loss(logits, labels)
        else:
            resized_labels = resize(input=labels, size=logits.shape[2:], mode='nearest', warning=False)
            return self.wrapped_loss(logits, resized_labels)

class CrossEntropy2D(nn.Module):
    def __init__(self, n_classes, reduction):
        super(CrossEntropy2D, self).__init__()

        self.n_classes = n_classes
        self.criteria = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, logits, labels):
        return self.criteria(logits.view(logits.shape[0], self.n_classes, -1), labels.view(labels.shape[0], -1))


class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float))
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        if self.thresh.device != logits.device:
            self.thresh = self.thresh.to(logits.device)
        loss = self.criteria(logits, labels.to(torch.long)).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss > self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)


class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):
        scores = F.softmax(logits, dim=1)
        factor = torch.pow(1.-scores, self.gamma)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss

