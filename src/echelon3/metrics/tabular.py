"""Metrics for tabular tasks (binary classification/scoring), in the
``echelon3.metrics.base.Metric`` interface (update/compute/reset). Computed on the CPU via
scikit-learn/scipy — the fit/predict trainer calls ``update(scores, y)`` with full arrays
(positive-class probabilities), then ``compute()``.
"""
import numpy as np

from echelon3.metrics.base import Metric


def _pos_scores(pred):
    """Reduce the model output to a vector of positive-class scores."""
    p = np.asarray(pred)
    if p.ndim == 2 and p.shape[1] == 2:   # predict_proba of a binary task -> class-1 column
        return p[:, 1]
    if p.ndim == 2 and p.shape[1] == 1:
        return p[:, 0]
    return p.ravel()


class _TabularMetric(Metric):
    """Base: accumulates (scores, targets) and computes the metric in ``compute()``."""

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
    """ROC AUC (higher is better)."""

    def compute(self):
        from sklearn.metrics import roc_auc_score
        pred, true = self._arrays()
        return float(roc_auc_score(true, _pos_scores(pred)))


class Gini(AUC):
    """Gini = 2·AUC − 1 (higher is better)."""

    def compute(self):
        return 2.0 * super().compute() - 1.0


class KS(_TabularMetric):
    """Kolmogorov–Smirnov: the max gap between the score CDFs of the positive and negative
    classes (higher is better). A credit-scoring classic."""

    def compute(self):
        from scipy.stats import ks_2samp
        pred, true = self._arrays()
        s = _pos_scores(pred)
        true = np.asarray(true)
        return float(ks_2samp(s[true == 1], s[true == 0]).statistic)


class LogLoss(_TabularMetric):
    """Binary cross-entropy (lower is better)."""

    def compute(self):
        from sklearn.metrics import log_loss
        pred, true = self._arrays()
        return float(log_loss(true, _pos_scores(pred), labels=[0, 1]))


class Accuracy(_TabularMetric):
    """Fraction correct at threshold 0.5 (higher is better)."""

    def __init__(self, threshold=0.5):
        self.threshold = threshold
        super().__init__()

    def compute(self):
        from sklearn.metrics import accuracy_score
        pred, true = self._arrays()
        return float(accuracy_score(true, (_pos_scores(pred) > self.threshold).astype(int)))


# ------------------------------------------------------------------ regression
# ADMET endpoints are regression; the model returns predict (not proba). Metrics use the direct output.

class MAE(_TabularMetric):
    """Mean absolute error (lower is better)."""

    def compute(self):
        from sklearn.metrics import mean_absolute_error
        pred, true = self._arrays()
        return float(mean_absolute_error(true, np.asarray(pred).ravel()))


class RMSE(_TabularMetric):
    """Root mean squared error (lower is better)."""

    def compute(self):
        from sklearn.metrics import mean_squared_error
        pred, true = self._arrays()
        return float(mean_squared_error(true, np.asarray(pred).ravel()) ** 0.5)


class R2(_TabularMetric):
    """Coefficient of determination R² (higher is better)."""

    def compute(self):
        from sklearn.metrics import r2_score
        pred, true = self._arrays()
        return float(r2_score(true, np.asarray(pred).ravel()))


class SpearmanR(_TabularMetric):
    """Spearman rank correlation (higher is better) — a common metric on ADMET leaderboards."""

    def compute(self):
        from scipy.stats import spearmanr
        pred, true = self._arrays()
        return float(spearmanr(np.asarray(pred).ravel(), np.asarray(true).ravel()).statistic)


class PearsonR(_TabularMetric):
    """Pearson correlation (higher is better)."""

    def compute(self):
        from scipy.stats import pearsonr
        pred, true = self._arrays()
        return float(pearsonr(np.asarray(pred).ravel(), np.asarray(true).ravel())[0])
