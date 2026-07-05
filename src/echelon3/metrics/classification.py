import torch
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from scipy.special import softmax
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score

from torch import Tensor
import torch.nn as nn

from echelon3.metrics.base import Metric


class EER(Metric):

    predicted_list = None
    target_list = None

    def __init__(self):
        self.reset()

    def update(self, predicted: torch.Tensor, target: torch.Tensor):
        self.predicted_list += predicted.cpu().numpy().tolist()
        self.target_list += target.cpu().numpy().tolist()

    def compute(self):
        trg = np.array(self.target_list)
        prd = np.array(self.predicted_list)
        if len(prd.shape) > 1:
            if prd.shape[1] == 2:
                prd = softmax(prd, axis=1)[:, 1]
            else:
                prd = np.squeeze(prd)
        fpr, tpr, thresholds = roc_curve(trg, prd, pos_label=1)
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        return eer

    def reset(self):
        self.predicted_list = []
        self.target_list = []


class AUC(Metric):

    predicted_list = None
    target_list = None

    def __init__(self):
        self.reset()

    def update(self, predicted: torch.Tensor, target: torch.Tensor):
        self.predicted_list += predicted.cpu().numpy().tolist()
        self.target_list += target.cpu().numpy().tolist()

    def compute(self):
        trg = np.array(self.target_list)
        prd = np.array(self.predicted_list)
        if len(prd.shape) > 1:
            if prd.shape[1] == 2:
                prd = softmax(prd, axis=1)[:, 1]
            else:
                prd = np.squeeze(prd)
        auc = roc_auc_score(trg, prd)
        return auc

    def reset(self):
        self.predicted_list = []
        self.target_list = []



class FrrAtFar(Metric):

    predicted_list = None
    target_list = None
    at_far = None

    def __init__(self, at_far: float):
        self.at_far = at_far
        self.reset()

    def update(self, predicted: torch.Tensor, target: torch.Tensor):
        self.predicted_list += predicted.cpu().numpy().tolist()
        self.target_list += target.cpu().numpy().tolist()

    def compute(self):
        trg = np.array(self.target_list)
        prd = np.array(self.predicted_list)

        if len(prd.shape) > 1:
            if prd.shape[1] == 2:
                prd = softmax(prd, axis=1)[:, 1]
            else:
                prd = np.squeeze(prd)

        fpr, tpr, thresholds = roc_curve(trg, prd, pos_label=1)
        far = 1 - tpr
        frr = fpr
        return interp1d(far, frr)(self.at_far)

    def reset(self):
        self.predicted_list = []
        self.target_list = []


