"""Молекулярная фичеризация как ``feature_transform`` для табличного estimator-пути.

OpenADMET/ADMET по сути табличная задача: вход — SMILES, выход — свойство. Здесь SMILES
превращается в числовые признаки (RDKit-дескрипторы и/или Morgan-фингерпринты), после чего
работает обычный ``EstimatorTrainer`` (деревья/FM). Фичеризатор плагинится как
``feature_transform`` (fit_transform/transform), кладётся в inference-бандл и на инференсе
считает признаки из SMILES новых молекул. Требует ``rdkit``.
"""
import numpy as np


# ================================================================= 2D graph route
# SMILES -> паддинг-граф (атомные признаки + нормированная матрица смежности) для 2D GNN
# на СУЩЕСТВУЮЩЕМ SGD-Trainer'е (чистый torch, без torch_geometric/кастомного collate).

_GRAPH_ELEMENTS = [6, 7, 8, 9, 15, 16, 17, 35, 53]  # C N O F P S Cl Br I (+ «прочее»)
ATOM_FEATURE_DIM = len(_GRAPH_ELEMENTS) + 1 + 4      # one-hot элемента + степень/заряд/аромат/H


def _atom_features(atom):
    z = atom.GetAtomicNum()
    onehot = [1.0 if z == e else 0.0 for e in _GRAPH_ELEMENTS]
    onehot.append(1.0 if z not in _GRAPH_ELEMENTS else 0.0)
    return onehot + [atom.GetDegree() / 4.0, float(atom.GetFormalCharge()),
                     float(atom.GetIsAromatic()), atom.GetTotalNumHs() / 4.0]


def mol_to_graph(smiles, max_atoms):
    """SMILES -> (node_features[max_atoms, F], adj_norm[max_atoms, max_atoms], mask[max_atoms])
    как torch-тензоры. adj — со self-loops и симметричной нормировкой D^-1/2 A D^-1/2."""
    import torch
    from rdkit import Chem

    nf = torch.zeros(max_atoms, ATOM_FEATURE_DIM, dtype=torch.float32)
    adj = torch.zeros(max_atoms, max_atoms, dtype=torch.float32)
    mask = torch.zeros(max_atoms, dtype=torch.float32)

    mol = Chem.MolFromSmiles(str(smiles)) if smiles is not None else None
    if mol is not None:
        n = min(mol.GetNumAtoms(), max_atoms)
        for i in range(n):
            nf[i] = torch.tensor(_atom_features(mol.GetAtomWithIdx(i)), dtype=torch.float32)
            mask[i] = 1.0
        a = torch.eye(max_atoms)  # self-loops
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if i < max_atoms and j < max_atoms:
                a[i, j] = a[j, i] = 1.0
        a = a * mask.unsqueeze(0) * mask.unsqueeze(1) + torch.eye(max_atoms) * (1 - mask)
        deg = a.sum(1).clamp(min=1.0)
        dinv = deg.pow(-0.5)
        adj = dinv.unsqueeze(1) * a * dinv.unsqueeze(0)
    return nf, adj, mask


class MoleculeGraphDataset:
    """SMILES -> паддинг-граф для 2D GNN на SGD-пути. ``__getitem__`` -> ((nf, adj, mask), y).
    Одиночный таргет-регрессия; строки с NaN-таргетом отбрасываются. augment/preprocess
    принимаются ради контракта картиночного пути, но игнорируются (граф строится из SMILES)."""

    def __init__(self, target, smiles_column="smiles", max_atoms=64,
                 path=None, source="auto", frame=None, read_kwargs=None,
                 augment=None, preprocess=None, **source_kwargs):
        from echelon3.data.tabular import _read_file, _read_sql
        import pandas as pd

        read_kwargs = dict(read_kwargs or {})
        if frame is not None:
            df = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
        elif source == "sql":
            df = _read_sql(**source_kwargs, **read_kwargs)
        else:
            df = _read_file(path=path, source=source, **source_kwargs, **read_kwargs)

        self.target = target
        self.smiles_column = smiles_column
        self.max_atoms = int(max_atoms)
        if target in df.columns:
            df = df[~df[target].isna()].reset_index(drop=True)   # только размеченные строки
            self._y = df[target].to_numpy(dtype=float)
        else:
            self._y = None
        self._smiles = df[smiles_column].tolist()

    def __len__(self):
        return len(self._smiles)

    def __getitem__(self, i):
        import torch
        nf, adj, mask = mol_to_graph(self._smiles[i], self.max_atoms)
        y = torch.tensor(float(self._y[i]) if self._y is not None else 0.0, dtype=torch.float32)
        return (nf, adj, mask), y

    def __str__(self):
        return f"MoleculeGraphDataset(mols={len(self._smiles)}, target='{self.target}', max_atoms={self.max_atoms})"


class SmilesFeaturizer:
    """SMILES-колонка -> матрица числовых молекулярных признаков.

    Args (через ``config:``):
      * ``smiles_column``: имя колонки со SMILES (по умолчанию ``smiles``).
      * ``descriptors``: включить RDKit-дескрипторы (~200 физхим. свойств).
      * ``fingerprint``: включить Morgan (ECFP) фингерпринт.
      * ``fp_bits`` / ``fp_radius``: размер и радиус фингерпринта (по умолчанию 1024 / 2).
      * ``scale_descriptors``: StandardScaler на блоке дескрипторов (fit на train).

    Невалидные SMILES дают строку из нулей/NaN→0. Морган-биты — 0/1. Дескрипторы,
    выбросившие исключение, зануляются (``nan_to_num``), чтобы путь был устойчив к любому
    движку (деревья съедят и NaN, но LogReg/TabPFN — нет).
    """

    def __init__(self, smiles_column="smiles", descriptors=True, fingerprint=True,
                 fp_bits=1024, fp_radius=2, scale_descriptors=False):
        self.smiles_column = smiles_column
        self.descriptors = descriptors
        self.fingerprint = fingerprint
        self.fp_bits = int(fp_bits)
        self.fp_radius = int(fp_radius)
        self.scale_descriptors = scale_descriptors
        self._desc_fns = None
        self._fpgen = None
        self._scaler = None

    # RDKit-объекты (функции дескрипторов, генератор фингерпринтов) не пиклятся — держим
    # их вне сериализуемого состояния и пересобираем лениво (_prep) после загрузки бандла.
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_desc_fns"] = None
        state["_fpgen"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _prep(self):
        if self.descriptors and self._desc_fns is None:
            from rdkit.Chem import Descriptors
            self._desc_fns = list(Descriptors.descList)  # [(name, fn), ...]
        if self.fingerprint and self._fpgen is None:
            from rdkit.Chem import rdFingerprintGenerator
            self._fpgen = rdFingerprintGenerator.GetMorganGenerator(
                radius=self.fp_radius, fpSize=self.fp_bits)

    @property
    def feature_names(self):
        self._prep()
        names = []
        if self.descriptors:
            names += [n for n, _ in self._desc_fns]
        if self.fingerprint:
            names += [f"fp_{i}" for i in range(self.fp_bits)]
        return names

    def _one(self, smi):
        from rdkit import Chem
        mol = Chem.MolFromSmiles(str(smi)) if smi is not None else None
        feats = []
        if self.descriptors:
            if mol is None:
                feats.extend([0.0] * len(self._desc_fns))
            else:
                for _, fn in self._desc_fns:
                    try:
                        feats.append(float(fn(mol)))
                    except Exception:
                        feats.append(0.0)
        if self.fingerprint:
            if mol is None:
                feats.extend([0] * self.fp_bits)
            else:
                feats.extend(self._fpgen.GetFingerprintAsNumPy(mol).tolist())
        return feats

    def _featurize(self, X):
        smiles = X[self.smiles_column] if hasattr(X, "columns") else X
        arr = np.array([self._one(s) for s in smiles], dtype=float)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

    def _n_desc(self):
        return len(self._desc_fns) if self.descriptors else 0

    def fit_transform(self, X, y=None):
        self._prep()
        arr = self._featurize(X)
        if self.scale_descriptors and self._n_desc() > 0:
            from sklearn.preprocessing import StandardScaler
            self._scaler = StandardScaler()
            n = self._n_desc()
            arr[:, :n] = self._scaler.fit_transform(arr[:, :n])
        return arr

    def transform(self, X):
        self._prep()
        arr = self._featurize(X)
        if self._scaler is not None:
            n = self._n_desc()
            arr[:, :n] = self._scaler.transform(arr[:, :n])
        return arr
