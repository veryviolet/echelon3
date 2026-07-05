"""Parameter-group construction helpers for fine-tuning.

Three concerns:

* `freeze_by_patterns(net, patterns)` — set `requires_grad=False` on every
  parameter whose dotted name matches any of the supplied regexes. Used by
  the LP→FT linear-probe stage (`patterns: ["^backbone\\."]`) and by the
  dual-transfer recipe (`patterns: ["^backbone\\.conv_stem", "^backbone\\.blocks\\.0", ...]`).

* `head_only_params(net)` — convenience: returns just the head parameters,
  used by the linear-probe (LP) stage of LP→FT when we do NOT freeze the
  backbone but pass only the head's parameters to the optimizer. Equivalent
  effect, slightly different semantics: with `freeze_by_patterns` BN running
  stats still update in `train()` mode; with `head_only_params` they do
  unless the caller also calls `net.eval()` on the backbone.

* `build_param_groups(net, groups_cfg, default_lr)` — build a list of
  parameter-groups dicts of the form `[{"params": [...], "lr": <float>}, ...]`
  suitable for direct passing to `torch.optim.AdamW(...)`. `groups_cfg` is a
  list of `(pattern, lr_multiplier)` pairs evaluated in order; the first
  matching pattern wins per parameter. Parameters not matched by any
  pattern receive `default_lr`. This is the Auto-RGN-flavour layer-wise LR
  used by the "layer-wise" fine-tune family — but the LR multipliers must
  be picked statically (true Auto-RGN computes gradient-norm ratios online;
  we use a U-shaped static profile that approximates @chen2023metalr's
  empirical conclusion that low and high layers want larger LRs than the
  middle of the network).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Iterable, List, Tuple

import torch


def _match_any(name: str, patterns: Iterable[str]) -> bool:
    return any(re.search(p, name) is not None for p in patterns)


def freeze_by_patterns(net: torch.nn.Module, patterns: Iterable[str]) -> int:
    """Set requires_grad=False on params matching any of `patterns`.

    Returns the number of frozen tensors. No-op if `patterns` is empty.
    """
    patterns = list(patterns or [])
    if not patterns:
        return 0
    n_frozen = 0
    for name, p in net.named_parameters():
        if _match_any(name, patterns):
            p.requires_grad_(False)
            n_frozen += 1
    return n_frozen


def head_only_params(net: torch.nn.Module) -> List[torch.nn.Parameter]:
    """Return the list of parameters belonging to the head only.

    Looks for an attribute named `head` first; if absent, falls back to
    parameter-name regex `head\\.` and finally to `classifier\\.`.
    """
    if hasattr(net, "head") and isinstance(net.head, torch.nn.Module):
        return [p for p in net.head.parameters() if p.requires_grad or True]
    head_params: List[torch.nn.Parameter] = []
    for name, p in net.named_parameters():
        if re.search(r"^head\.|\.head\.", name) is not None:
            head_params.append(p)
    if head_params:
        return head_params
    return [
        p for n, p in net.named_parameters()
        if re.search(r"^classifier\.|\.classifier\.", n) is not None
    ]


def build_param_groups(
    net: torch.nn.Module,
    groups_cfg: Iterable[Tuple[str, float]] | Iterable[dict],
    default_lr: float,
) -> List[dict]:
    """Build an optimizer-ready param-group list.

    `groups_cfg` items are either (pattern, lr_multiplier) tuples or dicts
    `{"pattern": str, "lr_mult": float}`. The first pattern matching a
    parameter's dotted name wins. Parameters with `requires_grad=False`
    are dropped entirely so the optimizer does not iterate over them.
    """
    norm: List[Tuple[str, float]] = []
    for entry in groups_cfg or []:
        if isinstance(entry, Mapping):
            # dict / OmegaConf DictConfig
            pat = entry["pattern"]
            lr_mult = entry["lr_mult"] if "lr_mult" in entry else 1.0
            norm.append((str(pat), float(lr_mult)))
        else:
            # (pattern, lr_mult) tuple
            norm.append((str(entry[0]), float(entry[1])))

    # Bucket params by which pattern they hit first; the "default" bucket
    # catches anything no pattern matches.
    buckets: dict[str, List[torch.nn.Parameter]] = {p: [] for p, _ in norm}
    buckets["__default__"] = []
    lr_for_bucket: dict[str, float] = {p: default_lr * m for p, m in norm}
    lr_for_bucket["__default__"] = default_lr

    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        for pat, _ in norm:
            if re.search(pat, name) is not None:
                buckets[pat].append(p)
                break
        else:
            buckets["__default__"].append(p)

    groups: List[dict] = []
    for key, params in buckets.items():
        if not params:
            continue
        groups.append({"params": params, "lr": lr_for_bucket[key], "name": key})
    return groups
