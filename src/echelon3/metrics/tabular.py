"""Метрики для табличных задач (бинарная классификация/скоринг), в интерфейсе
``echelon3.metrics.base.Metric`` (update/compute/reset). Считаются на CPU через
scikit-learn/scipy — fit/predict-трейнер зовёт ``update(scores, y)`` полными массивами
(вероятности положительного класса), затем ``compute()``.
"""
import numpy as np

from echelon3.metrics.base import Metric


def _pos_scores(pred):
    """Приводим выход модели к вектору скоров положительного класса."""
    p = np.asarray(pred)
    if p.ndim == 2 and p.shape[1] == 2:   # predict_proba бинарной задачи -> колонка класса 1
        return p[:, 1]
    if p.ndim == 2 and p.shape[1] == 1:
        return p[:, 0]
    return p.ravel()


class _TabularMetric(Metric):
    """База: накапливает (scores, targets) и считает метрику в ``compute()``."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._pred = []
        self._true = []

    def update(self, predicted, target):
        self._pred.append(np.asarray(predicted))
        self._true.append(np.asarray(target))

    def _arrays(self):
        return np.concatenate(self._pred), np.concatenate(self._true)


class AUC(_TabularMetric):
    """ROC AUC (выше — лучше)."""

    def compute(self):
        from sklearn.metrics import roc_auc_score
        pred, true = self._arrays()
        return float(roc_auc_score(true, _pos_scores(pred)))


class Gini(AUC):
    """Gini = 2·AUC − 1 (выше — лучше)."""

    def compute(self):
        return 2.0 * super().compute() - 1.0


class KS(_TabularMetric):
    """Kolmogorov–Smirnov: макс. разрыв CDF скоров положительного и отрицательного
    классов (выше — лучше). Классика кредитного скоринга."""

    def compute(self):
        from scipy.stats import ks_2samp
        pred, true = self._arrays()
        s = _pos_scores(pred)
        true = np.asarray(true)
        return float(ks_2samp(s[true == 1], s[true == 0]).statistic)


class LogLoss(_TabularMetric):
    """Бинарная кросс-энтропия (ниже — лучше)."""

    def compute(self):
        from sklearn.metrics import log_loss
        pred, true = self._arrays()
        return float(log_loss(true, _pos_scores(pred), labels=[0, 1]))


class Accuracy(_TabularMetric):
    """Доля верных при пороге 0.5 (выше — лучше)."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        super().__init__()

    def compute(self):
        from sklearn.metrics import accuracy_score
        pred, true = self._arrays()
        return float(accuracy_score(true, (_pos_scores(pred) > self.threshold).astype(int)))
