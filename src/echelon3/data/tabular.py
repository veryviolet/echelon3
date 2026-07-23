"""Tabular datasets from DIFFERENT sources for fit/predict trainers
(``echelon3.trainers.estimator.EstimatorTrainer``) and, optionally, for the gradient
path.

Source (``source``): a file (csv/parquet/feather/json/tsv), ``sql`` (connection +
query), or a ready-made ``frame`` (DataFrame/dict). The list is extensible — just add a
loader. Unlike image datasets, augment/preprocess are NOT required: tabular features
are returned as-is (the whole ``(X, y)`` via :meth:`Xy` for fit/predict, row by row via
``__getitem__`` for the gradient path).
"""
import os

import pandas as pd
from torch.utils.data import Dataset


_FILE_READERS = {
    ".csv": pd.read_csv,
    ".tsv": pd.read_table,
    ".parquet": pd.read_parquet,
    ".feather": pd.read_feather,
    ".json": pd.read_json,
}


def _read_file(path=None, source="auto", **read_kwargs):
    if path is None:
        raise ValueError("TabularDataset: file source needs 'path'")
    ext = os.path.splitext(str(path))[1].lower() if source in ("auto", "file") \
        else ("." + str(source).lstrip("."))
    if ext not in _FILE_READERS:
        raise ValueError(
            f"TabularDataset: unsupported file source '{ext}'; "
            f"supported: {sorted(_FILE_READERS)} (or source='sql'/frame=...)"
        )
    return _FILE_READERS[ext](path, **read_kwargs)


def _read_sql(sql=None, table=None, connection=None, **read_kwargs):
    if connection is None:
        raise ValueError("TabularDataset(source='sql'): 'connection' (string or DBAPI) is required")
    con = connection
    if isinstance(connection, str):
        # connection string -> SQLAlchemy engine; a live DBAPI connection is passed as-is.
        try:
            from sqlalchemy import create_engine
        except ImportError as e:
            raise RuntimeError(
                "TabularDataset(source='sql') with a connection string needs SQLAlchemy "
                "(pip install sqlalchemy), or pass an already-open DBAPI connection object"
            ) from e
        con = create_engine(connection)
    query = sql or (f"SELECT * FROM {table}" if table else None)
    if query is None:
        raise ValueError("TabularDataset(source='sql'): provide a 'sql' query or a 'table' name")
    return pd.read_sql(query, con, **read_kwargs)


class TabularDataset(Dataset):
    """Reads a table from ``source`` and splits features from the target.

    Args (via ``config:``):
      * ``target``: target column name (required; may be absent in the test table for
        inference — then ``y`` = ``None``).
      * ``features``: list of feature columns (by default — all except ``target`` and ``drop``).
      * ``drop``: columns to drop from the features (ids and the like).
      * ``categorical``: names of categorical columns (passed to the model at fit time, if it supports them).
      * ``source``: ``auto`` (by file extension) | ``csv``/``parquet``/... | ``sql``.
      * file source: ``path`` (+ ``read_kwargs`` for the pandas reader).
      * ``source: sql``: ``sql`` (query) or ``table``, plus ``connection`` (string/DBAPI).
      * ``frame``: a ready-made DataFrame/dict (in-memory, bypassing the source).
    """

    def __init__(self, target, features=None, drop=None, categorical=None,
                 source="auto", frame=None, read_kwargs=None, **source_kwargs):
        read_kwargs = dict(read_kwargs or {})

        if frame is not None:
            df = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
        elif source == "sql":
            df = _read_sql(**source_kwargs, **read_kwargs)
        else:
            df = _read_file(source=source, **source_kwargs, **read_kwargs)

        # target — a string (single) or a list (multi-target, e.g. ADMET endpoints).
        self.targets = [target] if isinstance(target, str) else list(target)
        self.target = self.targets[0]
        drop = set(drop or [])
        if features is None:
            features = [c for c in df.columns if c not in self.targets and c not in drop]
        self.features = list(features)
        self.categorical = list(categorical or [])

        self._X = df[self.features].reset_index(drop=True)
        present = [t for t in self.targets if t in df.columns]
        self._Y = df[present].reset_index(drop=True) if present else None  # all targets (DataFrame)
        self._y = df[self.target].to_numpy() if self.target in df.columns else None  # first (compatibility)

    def Xy(self):
        """The whole dataset at once: ``(X: DataFrame, y: ndarray | None)`` — for fit/predict
        (y is the first target; for multi-target use :meth:`y_frame`)."""
        return self._X, self._y

    def y_frame(self):
        """DataFrame of all present targets (multi-target) — columns may
        contain NaN (ADMET data is sparse: not every molecule has all endpoints measured)."""
        return self._Y

    @property
    def feature_names(self):
        return self.features

    def __len__(self):
        return len(self._X)

    def __getitem__(self, i):
        import numpy as np
        row = self._X.iloc[i].to_numpy(dtype=np.float32)
        y = self._y[i] if self._y is not None else -1
        return row, y

    def __str__(self):
        return (f"TabularDataset(rows={len(self._X)}, features={len(self.features)}, "
                f"target='{self.target}', source-cats={len(self.categorical)})")


class TabularPreprocessor:
    """A declarative preprocessor for tabular features (a wrapper around sklearn's
    ``ColumnTransformer``) — so that switching the engine stays a change to ``model:`` ONLY,
    even when the data has categories/missing values while the engine (LogReg/XGBoost/TabPFN)
    wants numbers.

    Numeric columns: impute (+ optional scale). Categorical: impute + encode
    (onehot|ordinal). Columns are auto-detected by dtype, or specified explicitly
    (``numeric``/``categorical``). ``EstimatorTrainer`` fits the preprocessor on train and
    applies it to the test sets (no leakage), and stores it fitted in the inference bundle.
    """

    def __init__(self, numeric=None, categorical=None,
                 num_impute="median", scale=False,
                 cat_impute="most_frequent", encode="onehot"):
        self.numeric = list(numeric) if numeric is not None else None
        self.categorical = list(categorical) if categorical is not None else None
        self.num_impute, self.scale = num_impute, scale
        self.cat_impute, self.encode = cat_impute, encode
        self._ct = None

    def _build(self, X):
        from sklearn.compose import ColumnTransformer
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

        num = self.numeric if self.numeric is not None \
            else list(X.select_dtypes(include="number").columns)
        cat = self.categorical if self.categorical is not None \
            else [c for c in X.columns if c not in num]

        num_steps = [("impute", SimpleImputer(strategy=self.num_impute))]
        if self.scale:
            num_steps.append(("scale", StandardScaler()))
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False) \
            if self.encode == "onehot" \
            else OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        cat_steps = [("impute", SimpleImputer(strategy=self.cat_impute)), ("encode", enc)]

        transformers = []
        if num:
            transformers.append(("num", Pipeline(num_steps), num))
        if cat:
            transformers.append(("cat", Pipeline(cat_steps), cat))
        return ColumnTransformer(transformers, remainder="drop")

    def fit_transform(self, X, y=None):
        self._ct = self._build(X)
        return self._ct.fit_transform(X, y)

    def transform(self, X):
        if self._ct is None:
            raise RuntimeError("TabularPreprocessor.transform before fit_transform")
        return self._ct.transform(X)
