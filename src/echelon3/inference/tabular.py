"""Инференс табличной модели из self-contained бандла, сохранённого
``echelon3.trainers.estimator.EstimatorTrainer``.

Бандл (модель + имена признаков + target) кладётся тем же CheckpointManager в .tar.
Здесь — загрузка (файл .tar или директория target: берём последний checkpoint) и
предсказание на новых данных (DataFrame или путь к таблице через TabularDataset).
"""
import glob
import os

import numpy as np
import torch

from echelon3.trainers.estimator import CHECKPOINT_ESTIMATOR_KEYWORD


def load_bundle(path):
    """Грузит бандл из .tar-файла или из директории (последний checkpoint-*.tar)."""
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
    """Предсказание на новых данных.

    ``data`` — DataFrame, ndarray, или путь к таблице (тогда читаем через TabularDataset
    по сохранённым в бандле именам признаков). Single-target -> ndarray. Multi-target
    (бандл с ``models``) -> ``{target: ndarray}``. Классификатор -> вероятность класса 1,
    регрессор -> сырой predict. Сохранённый feature_transform (напр. SmilesFeaturizer)
    ре-применяется автоматически.
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
