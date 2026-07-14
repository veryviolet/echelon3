"""Компактный 2D message-passing GNN на молекулярных графах (чистый PyTorch, без
torch_geometric). Работает на СУЩЕСТВУЮЩЕМ SGD-`Trainer`: вход — паддинг-граф
``(node_features, adj_norm, mask)`` из ``echelon3.data.molecular.MoleculeGraphDataset``,
выход — предсказание (по умолчанию скаляр на молекулу для регрессии ADMET-эндпоинта).

Слой: ``H <- ReLU(A_norm · H · W)`` (adjacency со self-loops и симметричной нормировкой
готовится в датасете), затем masked-mean readout по атомам и MLP-голова.
"""
import torch
import torch.nn as nn

from echelon3.data.molecular import ATOM_FEATURE_DIM


class MolGCN(nn.Module):
    def __init__(self, in_dim: int = ATOM_FEATURE_DIM, hidden: int = 64,
                 layers: int = 3, out_dim: int = 1, dropout: float = 0.0):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden)
        self.convs = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, out_dim))
        self.out_dim = out_dim

    def forward(self, source):
        node_features, adj, mask = source
        h = torch.relu(self.in_proj(node_features))          # [B, N, hidden]
        for conv in self.convs:
            h = self.drop(torch.relu(conv(torch.bmm(adj, h))))  # A·H·W
        h = h * mask.unsqueeze(-1)                            # обнулить паддинг-атомы
        pooled = h.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # masked mean
        out = self.head(pooled)                              # [B, out_dim]
        return out.squeeze(-1) if self.out_dim == 1 else out
