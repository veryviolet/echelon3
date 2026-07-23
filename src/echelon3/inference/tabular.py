"""Inference for a tabular model from the self-contained bundle saved by
``echelon3.trainers.estimator.EstimatorTrainer``.

The bundle (model + feature names + target) is stored in a .tar by the same CheckpointManager.
Here: loading (a .tar file or a target directory: we take the latest checkpoint) and
prediction on new data (a DataFrame or a path to a table via TabularDataset).
"""
import glob
import os

import numpy as np
import torch

from echelon3.trainers.estimator import CHECKPOINT_ESTIMATOR_KEYWORD


def load_bundle(path):
    """Loads the bundle from a .tar file or from a directory (the latest checkpoint-*.tar)."""
    if os.path.isdir(path):
        files = glob.glob(os.path.join(path, "checkpoint-*.tar"))
        if not files:
            raise FileNotFoundError(f"no checkpoint-*.tar under {path}")
        path = max(files, key=lambda f: int("".join(filter(str.isdigit, os.path.basename(f)))))
    obj = torch.load(path, map_location="cpu", weights_only=False)
    bundle = obj.get(CHECKPOINT_ESTIMATOR_KEYWORD, obj) if isinstance(obj, dict) else obj
    if not isinstance(bundle, dict) or ("model" not in bundle and "models" not in bundle):
        raise ValueError(f"{path} is not an echelon3 estimator bundle")
    return bundle


def _predict_one(model, X):
    if hasattr(model, "predict_proba"):
        p = np.asarray(model.predict_proba(X))
        return p[:, 1] if (p.ndim == 2 and p.shape[1] == 2) else p
    return np.asarray(model.predict(X))


def predict(bundle, data):
    """Prediction on new data.

    ``data`` — a DataFrame, ndarray, or a path to a table (in which case we read it via
    TabularDataset using the feature names stored in the bundle). Single-target -> ndarray.
    Multi-target (a bundle with ``models``) -> ``{target: ndarray}``. Classifier -> probability
    of class 1, regressor -> raw predict. A stored feature_transform (e.g. SmilesFeaturizer)
    is re-applied automatically.
    """
    features = bundle.get("features") or []
    if isinstance(data, str):
        from echelon3.data.tabular import TabularDataset
        tgt = bundle.get("targets") or bundle.get("target") or "__none__"
        ds = TabularDataset(target=tgt, features=features or None, path=data)
        X, _ = ds.Xy()
    else:
        import pandas as pd
        X = data[features] if (features and isinstance(data, pd.DataFrame)) else data

    ft = bundle.get("feature_transform")
    if ft is not None:
        X = ft.transform(X)

    if "models" in bundle:  # multi-target
        return {t: _predict_one(m, X) for t, m in bundle["models"].items()}
    return _predict_one(bundle["model"], X)
