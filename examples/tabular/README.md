# Tabular (fit/predict) example

The **estimator** path of echelon3: fit/predict models (trees, sklearn estimators,
tabular foundation models) configured in the same `module/type/config` idiom, driven by
`echelon3 train`. No `optimizer`/`loss`/`dataloaders` — the objective/loss is a
hyperparameter of the model itself (e.g. CatBoost `loss_function`).

## Run

```bash
python gen_tabular_data.py --root ./tab_data          # synthetic credit-scoring-like data
echelon3 train -cd . -cn credit_hgb device=cpu        # sklearn HistGradientBoosting (no extra deps)
```

## Swap the engine — change only the `model:` block

`credit_base.yaml` holds data/metrics/trainer/target; each engine is a tiny config that
composes it via `defaults:` and sets only `model:`:

```bash
echelon3 train -cd . -cn credit_lgbm      # LightGBM      (pip install lightgbm)
echelon3 train -cd . -cn credit_xgb       # XGBoost       (pip install xgboost)
echelon3 train -cd . -cn credit_rf        # sklearn RandomForest
echelon3 train -cd . -cn credit_catboost  # CatBoost      (pip install catboost)
echelon3 train -cd . -cn credit_logreg_ft # LogReg + feature_transform (impute/scale/encode)
```

## Feature preprocessing (`feature_transform`)

Engines that need numeric/scaled input (LogReg, TabPFN, …) or data with
categoricals/NaN use a declarative `feature_transform:` (`echelon3.data.tabular`
`TabularPreprocessor` — a sklearn `ColumnTransformer`). It is fit on train, applied to
test (no leakage) and saved into the inference bundle, so swapping engines stays a
change of only `model:`.

## Data sources

`TabularDataset` reads csv/parquet/feather/json/tsv, a **SQL** connection
(`source: sql`, `sql:`/`table:` + `connection:`), or an in-memory `frame:`.

## Inference

Each run saves a self-contained bundle (model + feature pipeline + feature names +
target) under `target.path`. Load and predict:

```python
from echelon3.inference.tabular import load_bundle, predict
bundle = load_bundle("./out_tab")          # dir (latest) or a checkpoint-*.tar
scores = predict(bundle, "new_rows.csv")   # DataFrame, ndarray, or a table path
```
