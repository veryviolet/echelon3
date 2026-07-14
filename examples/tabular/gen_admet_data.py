"""Синтетические ADMET-данные (OpenADMET-подобные): SMILES + несколько регрессионных
эндпоинтов, выведенных из реальных RDKit-дескрипторов (значит, обучаемы), с разреженностью
(часть значений NaN — как в реальных ADMET-кампаниях не у всех молекул измерены все
эндпоинты). Требует rdkit. Пишет train.csv/test.csv со столбцами
smiles, LogD, KSOL, HLM_CLint.
"""
import argparse
import os

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen

SMILES_POOL = [
    "CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "CC(=O)Nc1ccc(O)cc1", "c1ccccc1", "CCO", "Cc1ccccc1", "Oc1ccccc1", "OCC1OC(O)C(O)C(O)C1O",
    "CC(N)C(=O)O", "OC(=O)c1ccccc1", "Clc1ccccc1", "Nc1ccccc1", "COc1ccccc1", "CCN(CC)CC",
    "O=C(O)CCc1ccccc1", "Cc1ccc(cc1)S(=O)(=O)N", "CC(C)NCC(O)c1ccc(O)c(O)c1", "CN(C)CCc1c[nH]c2ccccc12",
    "c1ccc2ccccc2c1", "OCCO", "CC(=O)O", "CCOC(=O)C", "CC#N", "C1CCCCC1", "c1ccncc1",
    "c1ccc(cc1)c1ccccc1", "O=C(N)c1ccccc1", "CC(C)(C)O", "CCCCCCCC", "OC(=O)c1ccc(O)cc1",
    "Nc1ccc(cc1)S(=O)(=O)N", "CC1=CC(=O)CC(C)(C)C1", "Cc1ccc(cc1)C(=O)O", "COc1ccc(cc1)CC(N)C(=O)O",
    "Oc1ccc(cc1)C(=O)O", "CCc1ccccc1", "CN1CCC[C@H]1c1cccnc1", "Clc1ccc(Cl)cc1", "Fc1ccccc1",
    "Brc1ccccc1", "O=[N+]([O-])c1ccccc1", "NC(=O)c1ccncc1", "CC(O)C(=O)O", "OCc1ccccc1",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./admet_data")
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--sparsity", type=float, default=0.15)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    smis = rng.choice(SMILES_POOL, size=args.n)
    rows = []
    for smi in smis:
        mol = Chem.MolFromSmiles(smi)
        logp = Crippen.MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        tpsa = Descriptors.TPSA(mol)
        rows.append({
            "smiles": smi,
            "LogD": logp - 0.2 + rng.normal(0, 0.3),          # ~ lipophilicity
            "KSOL": -0.5 * logp + 2.0 + rng.normal(0, 0.4),   # solubility ~ -logp
            "HLM_CLint": 0.008 * mw + 0.02 * tpsa + rng.normal(0, 0.5),
        })
    df = pd.DataFrame(rows)
    for col in ["LogD", "KSOL", "HLM_CLint"]:                 # разреженность
        df.loc[rng.random(len(df)) < args.sparsity, col] = np.nan

    os.makedirs(args.root, exist_ok=True)
    split = int(args.n * 0.8)
    df.iloc[:split].to_csv(os.path.join(args.root, "train.csv"), index=False)
    df.iloc[split:].to_csv(os.path.join(args.root, "test.csv"), index=False)
    print(f"wrote {split} train / {args.n - split} test rows to {args.root} "
          f"(targets LogD/KSOL/HLM_CLint, ~{args.sparsity:.0%} missing each)")


if __name__ == "__main__":
    main()
