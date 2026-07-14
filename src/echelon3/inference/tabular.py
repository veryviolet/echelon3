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
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(f"{path} is not an echelon3 estimator bundle")
    return bundle


def predict(bundle, data):
    """Предсказание на новых данных.

    ``data`` — DataFrame, ndarray, или путь к таблице (тогда читаем через TabularDataset
    по сохранённым в бандле именам признаков). Возвращает вероятность положительного
    класса, если модель это умеет, иначе сырой ``predict``.
    """
    features = bundle.get("features") or []
    if isinstance(data, str):
        from echelon3.data.tabular import TabularDataset
        ds = TabularDataset(target=bundle.get("target") or "__none__",
                            features=features or None, path=data)
        X, _ = ds.Xy()
    else:
        import pandas as pd
        X = data[features] if (features and isinstance(data, pd.DataFrame)) else data

    ft = bundle.get("feature_transform")
    if ft is not None:
        X = ft.transform(X)

    model = bundle["model"]
    if bundle.get("predict_proba") and hasattr(model, "predict_proba"):
        p = np.asarray(model.predict_proba(X))
        return p[:, 1] if (p.ndim == 2 and p.shape[1] == 2) else p
    return np.asarray(model.predict(X))
