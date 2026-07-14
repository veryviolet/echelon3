"""OA-2: 2D GNN на SGD-пути echelon3 — SMILES -> паддинг-граф -> MolGCN -> baseline.Trainer.
Проверяет форму графа/выхода и что модель учится через штатный Trainer (checkpoint пишется).
rdkit — по importorskip; чистый torch (без torch_geometric)."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("rdkit")
import torch
from torch.utils.data import DataLoader

from echelon3.data.molecular import MoleculeGraphDataset, ATOM_FEATURE_DIM
from echelon3.nets.mol_gcn import MolGCN

SMIS = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "Cc1ccccc1", "Oc1ccccc1", "CCN(CC)CC",
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "c1ccncc1", "COc1ccccc1", "CCc1ccccc1", "Clc1ccccc1",
        "OCc1ccccc1", "Nc1ccccc1", "O=C(O)c1ccccc1", "CCCCCCCC", "c1ccc2ccccc2c1"]


def _frame(n, seed):
    from rdkit import Chem
    from rdkit.Chem import Crippen
    rng = np.random.default_rng(seed)
    smis = rng.choice(SMIS, size=n)
    y = [Crippen.MolLogP(Chem.MolFromSmiles(s)) + rng.normal(0, 0.2) for s in smis]
    return pd.DataFrame({"smiles": smis, "LogD": y})


def test_graph_dataset_and_net_shapes():
    ds = MoleculeGraphDataset(target="LogD", frame=_frame(20, 0), max_atoms=48)
    (nf, adj, mask), y = ds[0]
    assert nf.shape == (48, ATOM_FEATURE_DIM) and adj.shape == (48, 48) and mask.shape == (48,)
    assert y.ndim == 0
    (bnf, badj, bmask), by = next(iter(DataLoader(ds, batch_size=8)))
    out = MolGCN(hidden=32, layers=2)((bnf, badj, bmask))
    assert out.shape == (8,)                         # скаляр на молекулу


def test_gnn_trains_via_baseline_trainer(tmp_path):
    import torchmetrics
    from echelon3.trainers.baseline import Trainer
    from echelon3.checkpoint.manager import CheckpointManager

    net = MolGCN(hidden=48, layers=3)
    trainer = Trainer(
        epochs=8,
        train_dataloader=DataLoader(MoleculeGraphDataset(target="LogD", frame=_frame(160, 0)),
                                    batch_size=16, shuffle=True, drop_last=True),
        test_dataloader=DataLoader(MoleculeGraphDataset(target="LogD", frame=_frame(80, 1)),
                                   batch_size=32),
        net=net,
        losses={"mse": (torch.nn.MSELoss(), 1.0)},
        metrics={"r2": torchmetrics.R2Score()},
        optimizer=torch.optim.AdamW(net.parameters(), lr=0.003),
        scheduler=None,
        ckpt_manager=CheckpointManager(path=str(tmp_path), checkpoints_to_keep=1),
        mlops_logger=None,
        device=torch.device("cpu"),
        keep_best_on={"r2": {"mode": "directional", "value": "high"}},
        times_to_validate_per_epoch=1,
    )
    trainer.train()                                   # штатный SGD-путь, граф как «картинка»
    assert len(trainer._ckpt_manager.idxs) >= 1       # чекпойнт(ы) сохранены
