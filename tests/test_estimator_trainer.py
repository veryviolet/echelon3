"""P0: fit/predict-путь (EstimatorTrainer) — таблица -> fit -> eval -> save-бандл ->
инференс из бандла. Модель — sklearn (без внешних зависимостей)."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from echelon3.data.tabular import TabularDataset
from echelon3.metrics.tabular import AUC, Gini, KS, LogLoss
from echelon3.trainers.estimator import EstimatorTrainer
from echelon3.checkpoint.manager import CheckpointManager
from echelon3.inference.tabular import load_bundle, predict


def _frame(n, seed):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-(1.5 * x1 - 1.0 * x2)))
    y = (rng.random(n) < p).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def test_estimator_end_to_end(tmp_path):
    train = TabularDataset(target="y", frame=_frame(600, 0))
    test = TabularDataset(target="y", frame=_frame(300, 1))
    assert train.feature_names == ["x1", "x2"]  # target excluded

    trainer = EstimatorTrainer(
        model=LogisticRegression(),
        train_data=train, test_data=test,
        metrics={"auc": AUC(), "gini": Gini(), "ks": KS(), "ll": LogLoss()},
        ckpt_manager=CheckpointManager(path=str(tmp_path), checkpoints_to_keep=1),
        keep_best_on=["auc"],
    )
    results = trainer.train()

    assert "test" in results
    assert 0.5 < results["test"]["auc"] <= 1.0        # separable signal -> better than chance
    assert abs(results["test"]["gini"] - (2 * results["test"]["auc"] - 1)) < 1e-9

    # requirement #3: saved bundle is a self-contained inference artifact
    bundle = load_bundle(str(tmp_path))
    assert bundle["features"] == ["x1", "x2"] and bundle["target"] == "y"
    preds = predict(bundle, _frame(40, 2))
    assert preds.shape[0] == 40 and preds.min() >= 0.0 and preds.max() <= 1.0


def _frame_cat(n, seed):
    """Категориальная строковая колонка + пропуски — сырьё, которое LogReg не съест."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    home = rng.choice(["own", "rent", "mortgage"], size=n)
    bump = np.where(home == "own", 0.8, np.where(home == "rent", -0.5, 0.2))
    p = 1.0 / (1.0 + np.exp(-(1.2 * x1 + bump)))
    y = (rng.random(n) < p).astype(int)
    x1 = x1.copy()
    x1[::37] = np.nan  # пропуски
    return pd.DataFrame({"x1": x1, "home": home, "y": y})


def test_feature_transform_enables_categorical(tmp_path):
    from echelon3.data.tabular import TabularPreprocessor
    trainer = EstimatorTrainer(
        model=LogisticRegression(max_iter=200),
        train_data=TabularDataset(target="y", frame=_frame_cat(600, 0)),
        test_data=TabularDataset(target="y", frame=_frame_cat(300, 1)),
        metrics={"auc": AUC()},
        ckpt_manager=CheckpointManager(path=str(tmp_path), checkpoints_to_keep=1),
        feature_transform=TabularPreprocessor(scale=True),   # impute+scale num, impute+onehot cat
    )
    results = trainer.train()   # без feature_transform упало бы (строки + NaN)
    assert results["test"]["auc"] > 0.5

    # инференс из бандла ре-применяет тот же зафиченный трансформ
    bundle = load_bundle(str(tmp_path))
    assert bundle["feature_transform"] is not None
    preds = predict(bundle, _frame_cat(50, 2))
    assert preds.shape[0] == 50 and 0.0 <= preds.min() and preds.max() <= 1.0


def test_multitarget_all_nan_target_skipped(tmp_path):
    """M1: таргет со всеми-NaN в train не должен ронять весь ран — пропускаем его."""
    import pytest
    from sklearn.linear_model import LinearRegression
    from echelon3.trainers.estimator import MultiTargetEstimatorTrainer
    from echelon3.metrics.tabular import MAE

    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    df["ta"] = df["x1"] + rng.normal(0, 0.1, n)
    df["tb"] = np.nan                                   # эндпоинт без измерений
    trainer = MultiTargetEstimatorTrainer(
        model=LinearRegression(),
        train_data=TabularDataset(target=["ta", "tb"], frame=df),
        test_data=TabularDataset(target=["ta", "tb"], frame=df.copy()),
        metrics={"mae": MAE()},
        ckpt_manager=CheckpointManager(path=str(tmp_path), checkpoints_to_keep=1),
    )
    trainer.train()                                     # не падает на tb
    assert set(load_bundle(str(tmp_path))["models"]) == {"ta"}   # tb пропущен


def test_estimator_rejects_multitarget(tmp_path):
    """m2: одиночный EstimatorTrainer не должен молча учить только первый из списка таргетов."""
    import pytest
    df = pd.DataFrame({"x": np.arange(20.0), "y1": np.arange(20.0), "y2": np.arange(20.0)})
    trainer = EstimatorTrainer(
        model=LogisticRegression(),
        train_data=TabularDataset(target=["y1", "y2"], frame=df),
        test_data=None, metrics={},
        ckpt_manager=CheckpointManager(path=str(tmp_path), checkpoints_to_keep=1),
    )
    with pytest.raises(ValueError):
        trainer.train()


def test_multi_test_sets(tmp_path):
    trainer = EstimatorTrainer(
        model=LogisticRegression(),
        train_data=TabularDataset(target="y", frame=_frame(400, 0)),
        test_data={"a": TabularDataset(target="y", frame=_frame(150, 1)),
                   "b": TabularDataset(target="y", frame=_frame(150, 2))},
        metrics={"auc": AUC()},
        ckpt_manager=CheckpointManager(path=str(tmp_path), checkpoints_to_keep=1),
    )
    results = trainer.train()
    assert set(results) == {"a", "b"}
