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
import copy
import os

from omegaconf import DictConfig, OmegaConf, open_dict


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


def _resolve_path(config_name: str, config_dir: str) -> str:
    path = config_name if os.path.isabs(config_name) else os.path.join(config_dir, config_name)
    if not path.endswith((".yaml", ".yml")):
        path += ".yaml"
    return path


def _apply_default_entry(entry, config_dir, result, own, seen):
    """Обработать один элемент списка ``defaults``; вернуть (новый result, был_ли _self_).

    Формы:
      * ``_self_``            — подмешать собственное содержимое файла;
      * ``name`` / ``a/b``    — базовый конфиг (мёрдж в КОРЕНЬ, рекурсивно);
      * ``{group: option}``   — config-group: ``group/option.yaml`` под ключ ``group``
                                (вложенные ``a/b: opt`` → под ``a.b``), дефолтная
                                упаковка Hydra.
    """
    if entry == "_self_":
        return OmegaConf.merge(result, own), True
    if isinstance(entry, str):
        base = _load_with_defaults(entry, config_dir, seen)
        return OmegaConf.merge(result, base), False
    if isinstance(entry, (dict, DictConfig)):
        for group, option in entry.items():
            if option is None:  # '- group: null' — пропуск (без опции)
                continue
            sub = _load_with_defaults(f"{group}/{option}", config_dir, seen)
            container = OmegaConf.create({})
            OmegaConf.update(container, str(group).replace("/", "."), sub, force_add=True)
            result = OmegaConf.merge(result, container)
        return result, False
    raise ValueError(f"unsupported 'defaults' entry {entry!r}")


def _load_with_defaults(config_name: str, config_dir: str, _seen=None):
    """Загрузить YAML и, если есть Hydra-подобный ``defaults:``, скомпоновать конфиги.

    Слияние идёт СЛЕВА-НАПРАВО (следующий переопределяет предыдущего); ``_self_`` —
    место, где подмешивается собственное содержимое файла (если его нет в списке —
    неявно последним, как в Hydra). Поддержаны имена базовых конфигов (в т.ч. с
    подкаталогом) и config-groups ``{group: option}``; ``@package``-пути и
    ``override``/``optional`` в defaults не поддерживаются."""
    _seen = _seen or set()
    path = _resolve_path(config_name, config_dir)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"config not found: {path}")
    if path in _seen:
        raise ValueError(f"circular 'defaults' composition at {path}")
    _seen = _seen | {path}

    raw = OmegaConf.load(path)
    defaults = raw.get("defaults", None) if isinstance(raw, DictConfig) else None

    own = copy.deepcopy(raw)  # собственное содержимое файла (без defaults/hydra)
    with open_dict(own):
        own.pop("defaults", None)
        own.pop("hydra", None)

    if not defaults:
        return own

    result = OmegaConf.create({})
    self_included = False
    for entry in defaults:
        result, was_self = _apply_default_entry(entry, config_dir, result, own, _seen)
        self_included = self_included or was_self
    if not self_included:  # Hydra неявно ставит _self_ последним
        result = OmegaConf.merge(result, own)
    return result


def load_config(config_name: str, config_dir: str = ".", overrides=()):
    """Загрузить конфиг (с композицией ``defaults:``), отбросить ``hydra:`` и применить
    оверрайды.

    ``config_name`` — имя файла (с .yaml или без) либо абсолютный путь. ``config_dir`` —
    каталог поиска (по умолчанию текущий), как ``--config-dir`` у Hydra.
    """
    cfg = _load_with_defaults(config_name, config_dir)
    if isinstance(cfg, DictConfig) and "hydra" in cfg:
        with open_dict(cfg):
            del cfg["hydra"]
    apply_overrides(cfg, overrides)
    return cfg
