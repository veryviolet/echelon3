"""Fit/predict-трейнер для табличных моделей (деревья CatBoost/XGBoost/LightGBM,
табличные foundation-модели TabPFN/TabICL/TabFM/TabGPT — все sklearn-совместимые
in-context/fit-predict).

Это ОТДЕЛЬНОЕ семейство трейнеров: оно НЕ наследует картиночный
``echelon3.trainers.baseline.Trainer`` и не использует его SGD-машинерию — нет
optimizer/loss/scheduler/DataLoader/эпох. Objective/loss у таких моделей — это
ГИПЕРПАРАМЕТР самой модели (``model.config``, напр. CatBoost ``loss_function``), а не
отдельная секция ``loss:``.

Жизненный цикл: взять весь ``(X, y)`` из train-датасета → ``model.fit`` → предсказать и
посчитать метрики на каждом test-датасете → сохранить self-contained inference-бандл
(модель + имена признаков + target) тем же CheckpointManager (.tar), что и остальной
echelon3. Бандл грузится для инференса/экспорта (см. ``echelon3.inference.tabular``).
"""
import numpy as np
from colorama import Fore

CHECKPOINT_ESTIMATOR_KEYWORD = "estimator_bundle"


def _fmt(v) -> str:
    """Компактное число: до 4 знаков без хвостовых нулей, научная запись для крайних."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    a = abs(f)
    if a != 0.0 and (a < 1e-4 or a >= 1e6):
        return f"{f:.3e}"
    return f"{f:.4f}".rstrip("0").rstrip(".") or "0"


class EstimatorTrainer:
    """Обучение fit/predict-моделей в идеологии echelon3.

    Инъекции (даёт ``assemble_estimator`` в CLI): ``model``, ``train_data``,
    ``test_data`` (один датасет или dict именованных), ``metrics`` (dict), ``ckpt_manager``.
    Через ``config:``: ``keep_best_on`` (метрика для отчёта о лучшем), ``fit_kwargs``
    (доп. аргументы ``model.fit``), ``eval_set`` (bool — прокинуть первый test как eval_set
    в fit для ранней остановки), ``use_categorical`` (bool — прокинуть ``cat_features``).
    """

    def __init__(self, model, train_data, test_data, metrics, ckpt_manager,
                 feature_transform=None, keep_best_on=None, fit_kwargs=None,
                 eval_set=False, use_categorical=False, **kwargs):
        self.model = model
        self.feature_transform = feature_transform
        self.train_data = train_data
        if isinstance(test_data, dict):
            self.test_data = dict(test_data)
        elif test_data is not None:
            self.test_data = {"test": test_data}
        else:
            self.test_data = {}
        self.metrics = dict(metrics or {})
        self.ckpt = ckpt_manager
        self.keep_best_on = keep_best_on
        self.fit_kwargs = dict(fit_kwargs or {})
        self.eval_set = bool(eval_set)
        self.use_categorical = bool(use_categorical)

    # ------------------------------------------------------------------ predict
    def _scores(self, X, model=None):
        """Вероятность положительного класса (классификатор) или сырой predict
        (регрессор). model=None -> self.model."""
        model = model if model is not None else self.model
        if hasattr(model, "predict_proba"):
            p = np.asarray(model.predict_proba(X))
            return p[:, 1] if (p.ndim == 2 and p.shape[1] == 2) else p
        return np.asarray(model.predict(X))

    # -------------------------------------------------------------------- train
    def train(self):
        Xtr, ytr = self.train_data.Xy()
        if ytr is None:
            raise ValueError("EstimatorTrainer: train dataset has no target column to fit on")

        # Препроцессинг признаков: фитим на train, применяем к тестам (без утечки), кладём
        # зафиченным в бандл. Держит смену движка сменой только model: даже на категориях.
        if self.feature_transform is not None:
            print(f"--> Fitting feature_transform ({type(self.feature_transform).__name__})...")
            Xtr = self.feature_transform.fit_transform(Xtr, ytr)

        fit_kwargs = dict(self.fit_kwargs)
        if self.use_categorical and getattr(self.train_data, "categorical", None):
            fit_kwargs.setdefault("cat_features", list(self.train_data.categorical))
        if self.eval_set and self.test_data:
            first = next(iter(self.test_data.values()))
            Xv, yv = first.Xy()
            if yv is not None:
                fit_kwargs.setdefault("eval_set", [(Xv, yv)])

        n_feat = len(getattr(self.train_data, "feature_names", []) or [])
        print(f"--> Fitting {type(self.model).__name__} on {len(Xtr)} rows, {n_feat} source features...")
        try:
            self.model.fit(Xtr, ytr, **fit_kwargs)
        except TypeError:
            # модель не принимает наши extra fit-kwargs (eval_set/cat_features) — фитим голо
            self.model.fit(Xtr, ytr)
        print(Fore.LIGHTGREEN_EX + "--> Fitted." + Fore.CYAN)

        results = {}
        for name, ds in self.test_data.items():
            Xv, yv = ds.Xy()
            if yv is None:
                print(f"--> [{name}] has no target — skipping metrics")
                continue
            if self.feature_transform is not None:
                Xv = self.feature_transform.transform(Xv)
            scores = self._scores(Xv)
            vals = {}
            for mname, metric in self.metrics.items():
                metric.reset()
                metric.update(scores, yv)
                vals[mname] = metric.compute()
            results[name] = vals
            parts = ", ".join(f"{k}={_fmt(v)}" for k, v in vals.items())
            print(f"--> Evaluated [{name}]: {parts}")

        self._save(results)
        return results

    # --------------------------------------------------------------------- save
    def _save(self, results):
        """Self-contained inference-бандл (.tar через тот же CheckpointManager)."""
        self.ckpt.init_storage()
        bundle = {
            CHECKPOINT_ESTIMATOR_KEYWORD: {
                "framework": "echelon3.estimator",
                "model": self.model,
                "feature_transform": self.feature_transform,
                "features": list(getattr(self.train_data, "feature_names", [])),
                "categorical": list(getattr(self.train_data, "categorical", [])),
                "target": getattr(self.train_data, "target", None),
                "predict_proba": hasattr(self.model, "predict_proba"),
                "metrics": results,
            }
        }
        self.ckpt.save_checkpoint(bundle)
        best = ""
        if self.keep_best_on and results:
            first_set = next(iter(results.values()))
            tracked = {k: first_set[k] for k in self.keep_best_on if k in first_set} \
                if hasattr(self.keep_best_on, "__iter__") else {}
            if tracked:
                best = " (" + ", ".join(f"{k}={_fmt(v)}" for k, v in tracked.items()) + ")"
        print(Fore.LIGHTGREEN_EX + f"--> Saved inference bundle to {self.ckpt.path}{best}" + Fore.CYAN)

    def close(self):
        # нет воркеров/ресурсов — заглушка ради совместимости с CLI (trainer.close()).
        pass


class MultiTargetEstimatorTrainer(EstimatorTrainer):
    """Много таргетов за один ран (напр. 9 ADMET-эндпоинтов): на каждый target — своя
    копия модели (``sklearn.base.clone``), обучается на строках, где таргет измерен
    (NaN-маска; ADMET-данные разрежены). Метрики считаются per-target. Общий
    ``feature_transform`` (напр. SmilesFeaturizer) фитится ОДИН раз и переиспользуется.
    Бандл: ``{target: model}`` + общий feature_transform.

    Конфиг тот же, что у EstimatorTrainer, но ``data.*.config.target`` — СПИСОК эндпоинтов.
    """

    def train(self):
        from sklearn.base import clone

        X, _ = self.train_data.Xy()
        Y = self.train_data.y_frame()
        if Y is None:
            raise ValueError("MultiTargetEstimatorTrainer: no target columns to fit on")
        targets = list(getattr(self.train_data, "targets", [self.train_data.target]))

        if self.feature_transform is not None:
            print(f"--> Fitting feature_transform ({type(self.feature_transform).__name__})...")
            Xt = np.asarray(self.feature_transform.fit_transform(X, None))
        else:
            Xt = np.asarray(X)

        prepared = {}
        for name, ds in self.test_data.items():
            Xv, _ = ds.Xy()
            Xvt = self.feature_transform.transform(Xv) if self.feature_transform is not None else Xv
            prepared[name] = (np.asarray(Xvt), ds.y_frame())

        models, results = {}, {name: {} for name in self.test_data}
        for t in targets:
            yt = np.asarray(Y[t], dtype=float)
            mask = ~np.isnan(yt)
            model = clone(self.model)
            print(f"--> [{t}] fitting on {int(mask.sum())} labelled rows...")
            model.fit(Xt[mask], yt[mask])
            models[t] = model
            for name, (Xvt, Yv) in prepared.items():
                if Yv is None or t not in Yv:
                    continue
                yv = np.asarray(Yv[t], dtype=float)
                vmask = ~np.isnan(yv)
                if vmask.sum() == 0:
                    continue
                preds = self._scores(Xvt[vmask], model=model)
                vals = {}
                for mname, metric in self.metrics.items():
                    metric.reset()
                    metric.update(preds, yv[vmask])
                    vals[mname] = metric.compute()
                results[name][t] = vals
                parts = ", ".join(f"{k}={_fmt(v)}" for k, v in vals.items())
                print(f"--> Evaluated [{name}/{t}]: {parts}")

        self._save_multi(models, results, targets)
        return results

    def _save_multi(self, models, results, targets):
        self.ckpt.init_storage()
        bundle = {CHECKPOINT_ESTIMATOR_KEYWORD: {
            "framework": "echelon3.estimator.multitarget",
            "models": models,                       # {target: fitted model}
            "feature_transform": self.feature_transform,
            "features": list(getattr(self.train_data, "feature_names", [])),
            "targets": list(targets),
            "predict_proba": hasattr(self.model, "predict_proba"),
            "metrics": results,
        }}
        self.ckpt.save_checkpoint(bundle)
        print(Fore.LIGHTGREEN_EX
              + f"--> Saved multi-target bundle ({len(models)} targets) to {self.ckpt.path}"
              + Fore.CYAN)
