"""OA-1: OpenADMET как табличная задача — SMILES -> RDKit-фичи (feature_transform) ->
модель на эндпоинт (MultiTargetEstimatorTrainer), регрессионные метрики, разреженные
таргеты, инференс из мульти-таргет бандла. Модель — sklearn (без catboost); rdkit — по importorskip."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("rdkit")
from sklearn.ensemble import RandomForestRegressor

from echelon3.data.tabular import TabularDataset
from echelon3.data.molecular import SmilesFeaturizer
from echelon3.metrics.tabular import MAE, R2, SpearmanR
from echelon3.trainers.estimator import MultiTargetEstimatorTrainer
from echelon3.checkpoint.manager import CheckpointManager
from echelon3.inference.tabular import load_bundle, predict

SMIS = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "Cc1ccccc1", "Oc1ccccc1", "CCN(CC)CC",
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "c1ccncc1", "COc1ccccc1", "CCc1ccccc1", "Clc1ccccc1",
        "OCc1ccccc1", "Nc1ccccc1", "O=C(O)c1ccccc1", "CCCCCCCC", "c1ccc2ccccc2c1"]


def _frame(n, seed):
    from rdkit import Chem
    from rdkit.Chem import Crippen
    rng = np.random.default_rng(seed)
    smis = rng.choice(SMIS, size=n)
    rows = []
    for smi in smis:
        lp = Crippen.MolLogP(Chem.MolFromSmiles(smi))
        rows.append({"smiles": smi, "LogD": lp + rng.normal(0, 0.2),
                     "KSOL": -0.5 * lp + rng.normal(0, 0.2)})
    df = pd.DataFrame(rows)
    df.loc[rng.random(n) < 0.15, "KSOL"] = np.nan     # разреженность одного таргета
    return df


def test_smiles_featurizer_shapes():
    f = SmilesFeaturizer(fp_bits=64)
    arr = f.fit_transform(pd.DataFrame({"smiles": ["CCO", "not_a_smiles###", "c1ccccc1"]}))
    assert arr.shape == (3, len(f.feature_names))
    assert np.isfinite(arr).all()                     # nan_to_num -> устойчиво к битым SMILES


def test_multitarget_end_to_end(tmp_path):
    train = TabularDataset(target=["LogD", "KSOL"], frame=_frame(300, 0))
    test = TabularDataset(target=["LogD", "KSOL"], frame=_frame(150, 1))
    assert train.targets == ["LogD", "KSOL"]
    assert train.feature_names == ["smiles"]          # таргеты исключены из фич

    trainer = MultiTargetEstimatorTrainer(
        model=RandomForestRegressor(n_estimators=80, random_state=0),
        train_data=train, test_data=test,
        metrics={"mae": MAE(), "r2": R2(), "spearman": SpearmanR()},
        ckpt_manager=CheckpointManager(path=str(tmp_path), checkpoints_to_keep=1),
        feature_transform=SmilesFeaturizer(fp_bits=128),
    )
    results = trainer.train()
    assert set(results["test"]) == {"LogD", "KSOL"}
    assert results["test"]["LogD"]["r2"] > 0.3        # LogD ~ MolLogP -> обучаемо

    # инференс: мульти-таргет бандл -> {target: preds}, feature_transform ре-применён
    bundle = load_bundle(str(tmp_path))
    assert set(bundle["models"]) == {"LogD", "KSOL"} and bundle["targets"] == ["LogD", "KSOL"]
    preds = predict(bundle, _frame(20, 2))
    assert set(preds) == {"LogD", "KSOL"} and preds["LogD"].shape[0] == 20
