"""Загрузка конфига (OmegaConf) + Hydra-подобные CLI-оверрайды — без Hydra.

Оверрайды (позиционные аргументы CLI):

    key=value        — задать / переопределить (в т.ч. вложенный a.b.c=...)
    +key=value       — добавить (эквивалент задать; префикс — для совместимости команд)
    ++key=value      — добавить-или-переопределить
    ~key             — удалить ключ

Значения типизируются грамматикой OmegaConf: int / float / bool / null, списки
``[1,2,3]``, строки, интерполяции ``${oc.env:VAR,default}``. Оверрайды ``hydra.*``
игнорируются (no-op) — Hydra больше нет. В отличие от Hydra нет strict/struct-режима:
``key=value`` спокойно добавляет новый ключ (без обязательного ``+``).
"""
import os

from omegaconf import OmegaConf, open_dict


def _parse_value(raw: str):
    """Типизировать строковое значение оверрайда грамматикой OmegaConf."""
    return OmegaConf.from_dotlist([f"_ov_={raw}"])._ov_


def _delete_key(cfg, dotted: str):
    parent_path, _, leaf = dotted.rpartition(".")
    parent = OmegaConf.select(cfg, parent_path) if parent_path else cfg
    if parent is not None and leaf in parent:
        with open_dict(parent):
            del parent[leaf]


def apply_overrides(cfg, overrides):
    """Применить список строковых оверрайдов к OmegaConf-конфигу на месте."""
    for ov in overrides:
        if ov.startswith("~"):  # удаление: ~key (или ~key=... — значение игнорируем)
            _delete_key(cfg, ov[1:].split("=", 1)[0])
            continue
        if ov.startswith("++"):
            ov = ov[2:]
        elif ov.startswith("+"):
            ov = ov[1:]
        if "=" not in ov:
            raise ValueError(f"override must be 'key=value' or '~key': {ov!r}")
        key, raw = ov.split("=", 1)
        if key == "hydra" or key.startswith("hydra."):
            continue  # Hydra-only — no-op после ухода от Hydra
        OmegaConf.update(cfg, key, _parse_value(raw), force_add=True)
    return cfg


def load_config(config_name: str, config_dir: str = ".", overrides=()):
    """Загрузить YAML через OmegaConf, отбросить Hydra-only секции и применить оверрайды.

    ``config_name`` — имя файла (с .yaml или без) либо абсолютный путь. ``config_dir`` —
    каталог поиска (по умолчанию текущий), как ``--config-dir`` у Hydra.
    """
    path = config_name if os.path.isabs(config_name) else os.path.join(config_dir, config_name)
    if not path.endswith((".yaml", ".yml")):
        path += ".yaml"
    if not os.path.isfile(path):
        raise FileNotFoundError(f"config not found: {path}")
    cfg = OmegaConf.load(path)
    for k in ("hydra", "defaults"):  # секции Hydra (композиция/оутпут-дир) движку не нужны
        if k in cfg:
            with open_dict(cfg):
                del cfg[k]
    apply_overrides(cfg, overrides)
    return cfg
