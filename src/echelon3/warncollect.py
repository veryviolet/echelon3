"""Collecting warnings instead of printing them inline.

Inline warnings break tqdm progress bars and clutter the log. Instead we suppress
their printing, aggregate them by text, and print a short summary in the standard
echelon3 format (``--> ...``) before each validation. The module has no
dependencies on the rest of echelon3 — it is imported both from the CLI and from
the trainer."""
import warnings
from collections import Counter

_counts: Counter = Counter()
_installed = False


def install():
    """Redirect the display of warnings into a counter (idempotent)."""
    global _installed
    if _installed:
        return
    # "always" — so that every repeat reaches the counter (otherwise Python's default
    # dedup would show a warning once per call site and the count would be understated).
    # Then, as a prefix, we ignore known dev noise — it will not reach the counter.
    warnings.simplefilter("always")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    def _collect(message, category, filename, lineno, file=None, line=None):
        name = getattr(category, "__name__", str(category))
        first = str(message).splitlines()[0] if str(message) else ""
        text = (first[:117] + "…") if len(first) > 120 else first
        _counts[f"{name}: {text}"] += 1

    warnings.showwarning = _collect
    _installed = True


def flush(limit: int = 5):
    """Prints a short summary of the accumulated warnings (tqdm-safe) and clears them."""
    if not _counts:
        return
    try:
        from tqdm import tqdm
        emit = tqdm.write
    except Exception:
        emit = print
    items = _counts.most_common()
    total = sum(_counts.values())
    emit(f"--> {total} warning(s) since last report:")
    for key, n in items[:limit]:
        emit(f"      {n}x {key}")
    if len(items) > limit:
        emit(f"      ... +{len(items) - limit} more")
    _counts.clear()
