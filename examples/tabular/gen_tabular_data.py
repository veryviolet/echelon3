"""Синтетический табличный датасет кредитного-скоринга типа (бинарный дефолт) —
чтобы estimator-путь echelon3 крутился без внешних данных. Пишет train.csv/test.csv."""
import argparse
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./tab_data")
    ap.add_argument("--n", type=int, default=4000)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    n = args.n
    age = rng.integers(18, 80, n)
    income = rng.gamma(2.0, 20000.0, n)
    debt_ratio = rng.beta(2, 5, n)
    n_loans = rng.poisson(1.5, n)
    util = rng.beta(2, 3, n)

    logit = (-3.0 + 2.0 * util + 3.0 * debt_ratio
             - 1e-5 * income + 0.2 * n_loans - 0.01 * (age - 40))
    p = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < p).astype(int)

    df = pd.DataFrame({
        "age": age, "income": income, "debt_ratio": debt_ratio,
        "n_loans": n_loans, "util": util, "default": y,
    })
    os.makedirs(args.root, exist_ok=True)
    split = int(n * 0.8)
    df.iloc[:split].to_csv(os.path.join(args.root, "train.csv"), index=False)
    df.iloc[split:].to_csv(os.path.join(args.root, "test.csv"), index=False)
    print(f"wrote {split} train / {n - split} test rows to {args.root} "
          f"(default rate={y.mean():.3f})")


if __name__ == "__main__":
    main()
