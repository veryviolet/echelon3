"""Fit/predict trainer for tabular models (CatBoost/XGBoost/LightGBM trees,
tabular foundation models TabPFN/TabICL/TabFM/TabGPT — all sklearn-compatible
in-context/fit-predict).

This is a SEPARATE family of trainers: it does NOT inherit the image-oriented
``echelon3.trainers.baseline.Trainer`` and does not use its SGD machinery — there is no
optimizer/loss/scheduler/DataLoader/epochs. For such models the objective/loss is a
HYPERPARAMETER of the model itself (``model.config``, e.g. CatBoost ``loss_function``), not
a separate ``loss:`` section.

Lifecycle: take the whole ``(X, y)`` from the train dataset → ``model.fit`` → predict and
compute metrics on each test dataset → save a self-contained inference bundle
(model + feature names + target) with the same CheckpointManager (.tar) as the rest of
echelon3. The bundle is loaded for inference/export (see ``echelon3.inference.tabular``).
"""
import numpy as np
from colorama import Fore

CHECKPOINT_ESTIMATOR_KEYWORD = "estimator_bundle"


def _fmt(v) -> str:
    """Compact number: up to 4 digits with no trailing zeros, scientific notation for extremes."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    a = abs(f)
    if a != 0.0 and (a < 1e-4 or a >= 1e6):
        return f"{f:.3e}"
    return f"{f:.4f}".rstrip("0").rstrip(".") or "0"


class EstimatorTrainer:
    """Training of fit/predict models in the echelon3 style.

    Injections (provided by ``assemble_estimator`` in the CLI): ``model``, ``train_data``,
    ``test_data`` (a single dataset or a dict of named ones), ``metrics`` (dict), ``ckpt_manager``.
    Via ``config:``: ``keep_best_on`` (metric to report the best on), ``fit_kwargs``
    (extra arguments for ``model.fit``), ``eval_set`` (bool — pass the first test set as eval_set
    to fit for early stopping), ``use_categorical`` (bool — pass ``cat_features``).
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
        """Probability of the positive class (classifier) or raw predict
        (regressor). model=None -> self.model."""
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
        if len(getattr(self.train_data, "targets", []) or []) > 1:
            raise ValueError(
                "EstimatorTrainer received a multi-target dataset (target is a list) — use "
                "echelon3.trainers.estimator.MultiTargetEstimatorTrainer instead")

        # Feature preprocessing: fit on train, apply to the test sets (no leakage), store the
        # fitted transform in the bundle. Lets you switch engines by changing only model: even on categoricals.
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
            # The model did not accept the AUTO-added eval_set/cat_features — drop only those
            # (keep the user's config.fit_kwargs) and refit.
            stripped = {k: v for k, v in fit_kwargs.items() if k not in ("eval_set", "cat_features")}
            print("--> model.fit rejected eval_set/cat_features — refitting without them")
            self.model.fit(Xtr, ytr, **stripped)
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
        """Self-contained inference bundle (.tar via the same CheckpointManager)."""
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
            kb = self.keep_best_on
            if isinstance(kb, str):
                kb = [kb]
            elif hasattr(kb, "keys"):          # dict form of keep_best_on
                kb = list(kb.keys())
            tracked = {k: first_set[k] for k in (kb or []) if k in first_set}
            if tracked:
                best = " (" + ", ".join(f"{k}={_fmt(v)}" for k, v in tracked.items()) + ")"
        print(Fore.LIGHTGREEN_EX + f"--> Saved inference bundle to {self.ckpt.path}{best}" + Fore.CYAN)

    def close(self):
        # no workers/resources — a stub for CLI compatibility (trainer.close()).
        pass


class MultiTargetEstimatorTrainer(EstimatorTrainer):
    """Many targets in a single run (e.g. 9 ADMET endpoints): each target gets its own
    copy of the model (``sklearn.base.clone``), trained on the rows where the target is measured
    (NaN mask; ADMET data is sparse). Metrics are computed per target. A shared
    ``feature_transform`` (e.g. SmilesFeaturizer) is fitted ONCE and reused.
    Bundle: ``{target: model}`` + the shared feature_transform.

    Same config as EstimatorTrainer, but ``data.*.config.target`` is a LIST of endpoints.
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
            if mask.sum() == 0:
                print(f"--> [{t}] no labelled rows in train — skipping this target")
                continue
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
