"""Partial weight loader: грузим только совпадающие по имени И форме веса.

Использование для warm-restart с архитектурой-расширением (MultiBinarySegmenter
→ MultiBinarySegmenterPlus): backbone + neck.projs + heads.{road,water,...}
переносятся из v17 ckpt; D-LinkNet center block и aux головы инициализируются
случайно (как при scratch инициализации).
"""
from __future__ import annotations

import torch


class PartialWeightsLoader:
    """Load matching (key, shape) tensors from a checkpoint into a network,
    skipping mismatched/missing entries. Mirrors HuggingFace strict=False semantics."""

    def __init__(self, strip_prefix: str | None = None, **kwargs):
        # strip_prefix — снимает префикс из ключей чекпойнта (например 'module.'
        # после DDP wrap), если нужно. None — без обработки.
        self.strip_prefix = strip_prefix

    def load(self, net, weights, device):
        # weights_only=False — echelon3 чекпойнты сериализуют объекты trainer/metric
        # (numpy/echelon-классы), которых torch 2.6 strict-mode не разрешает.
        ckpt = torch.load(weights, map_location=device, weights_only=False)
        # Tar/dict-of-state форматы echelon3 могут быть:
        #   {'net': state_dict, ...}  /  {'state_dict': sd}  /  raw sd
        if isinstance(ckpt, dict):
            for k in ("model_state_dict", "net", "state_dict"):
                if k in ckpt and isinstance(ckpt[k], dict):
                    sd = ckpt[k]
                    break
            else:
                sd = ckpt
        else:
            sd = ckpt

        if self.strip_prefix:
            p = self.strip_prefix
            sd = {(k[len(p):] if k.startswith(p) else k): v for k, v in sd.items()}

        model_sd = net.state_dict()
        matched = {}
        skipped_shape = []
        skipped_missing = []
        for k, v in sd.items():
            if k not in model_sd:
                skipped_missing.append(k)
                continue
            if model_sd[k].shape != v.shape:
                skipped_shape.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
                continue
            matched[k] = v

        model_sd.update(matched)
        net.load_state_dict(model_sd, strict=False)
        n_total = len(model_sd)
        n_loaded = len(matched)
        print(f"[PartialWeightsLoader] loaded {n_loaded}/{n_total} keys")
        if skipped_missing:
            print(f"  skipped missing in net: {len(skipped_missing)} (first 3: {skipped_missing[:3]})")
        if skipped_shape:
            print(f"  skipped shape-mismatch: {len(skipped_shape)} (first 3: {skipped_shape[:3]})")
