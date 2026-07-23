"""Config loading (OmegaConf) + Hydra-like CLI overrides — without Hydra.

Overrides (positional CLI arguments):

    key=value        — set / override (including nested a.b.c=...)
    +key=value       — add (equivalent to set; the prefix is for command compatibility)
    ++key=value      — add-or-override
    ~key             — delete a key

Values are typed by the OmegaConf grammar: int / float / bool / null, lists
``[1,2,3]``, strings, interpolations ``${oc.env:VAR,default}``. ``hydra.*`` overrides
are ignored (no-op) — there is no more Hydra. Unlike Hydra there is no strict/struct mode:
``key=value`` happily adds a new key (no mandatory ``+``).
"""
import copy
import os

from omegaconf import DictConfig, OmegaConf, open_dict


def _parse_value(raw: str):
    """Type an override's string value using the OmegaConf grammar."""
    return OmegaConf.from_dotlist([f"_ov_={raw}"])._ov_


def _delete_key(cfg, dotted: str):
    parent_path, _, leaf = dotted.rpartition(".")
    parent = OmegaConf.select(cfg, parent_path) if parent_path else cfg
    if parent is not None and leaf in parent:
        with open_dict(parent):
            del parent[leaf]


def apply_overrides(cfg, overrides):
    """Apply a list of string overrides to an OmegaConf config in place."""
    for ov in overrides:
        if ov.startswith("~"):  # deletion: ~key (or ~key=... — the value is ignored)
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
            continue  # Hydra-only — no-op now that Hydra is gone
        OmegaConf.update(cfg, key, _parse_value(raw), force_add=True)
    return cfg


def _resolve_path(config_name: str, config_dir: str) -> str:
    path = config_name if os.path.isabs(config_name) else os.path.join(config_dir, config_name)
    if not path.endswith((".yaml", ".yml")):
        path += ".yaml"
    return path


def _apply_default_entry(entry, config_dir, result, own, seen):
    """Process one entry of the ``defaults`` list; return (new result, whether _self_ was seen).

    Forms:
      * ``_self_``            — mix in the file's own content;
      * ``name`` / ``a/b``    — a base config (merged into the ROOT, recursively);
      * ``{group: option}``   — a config-group: ``group/option.yaml`` under the ``group`` key
                                (nested ``a/b: opt`` → under ``a.b``), Hydra's default
                                packaging.
    """
    if entry == "_self_":
        return OmegaConf.merge(result, own), True
    if isinstance(entry, str):
        base = _load_with_defaults(entry, config_dir, seen)
        return OmegaConf.merge(result, base), False
    if isinstance(entry, (dict, DictConfig)):
        for group, option in entry.items():
            if option is None:  # '- group: null' — skip (no option)
                continue
            sub = _load_with_defaults(f"{group}/{option}", config_dir, seen)
            container = OmegaConf.create({})
            OmegaConf.update(container, str(group).replace("/", "."), sub, force_add=True)
            result = OmegaConf.merge(result, container)
        return result, False
    raise ValueError(f"unsupported 'defaults' entry {entry!r}")


def _load_with_defaults(config_name: str, config_dir: str, _seen=None):
    """Load a YAML and, if there is a Hydra-like ``defaults:``, compose the configs.

    Merging goes LEFT-TO-RIGHT (each next one overrides the previous); ``_self_`` is
    the place where the file's own content is mixed in (if it is not in the list —
    implicitly last, as in Hydra). Base config names (including ones in a
    subdirectory) and config-groups ``{group: option}`` are supported; ``@package``
    paths and ``override``/``optional`` in defaults are not supported."""
    _seen = _seen or set()
    path = _resolve_path(config_name, config_dir)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"config not found: {path}")
    if path in _seen:
        raise ValueError(f"circular 'defaults' composition at {path}")
    _seen = _seen | {path}

    raw = OmegaConf.load(path)
    defaults = raw.get("defaults", None) if isinstance(raw, DictConfig) else None

    own = copy.deepcopy(raw)  # the file's own content (without defaults/hydra)
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
    if not self_included:  # Hydra implicitly places _self_ last
        result = OmegaConf.merge(result, own)
    return result


def load_config(config_name: str, config_dir: str = ".", overrides=()):
    """Load a config (with ``defaults:`` composition), drop ``hydra:`` and apply the
    overrides.

    ``config_name`` — a file name (with or without .yaml) or an absolute path. ``config_dir`` —
    the search directory (the current one by default), like Hydra's ``--config-dir``.
    """
    cfg = _load_with_defaults(config_name, config_dir)
    if isinstance(cfg, DictConfig) and "hydra" in cfg:
        with open_dict(cfg):
            del cfg["hydra"]
    apply_overrides(cfg, overrides)
    return cfg
