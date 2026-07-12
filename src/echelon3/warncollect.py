"""Сбор предупреждений вместо инлайн-печати.

Инлайн-варнинги рвут tqdm-прогрессбары и засоряют лог. Вместо этого глушим их
печать, агрегируем по тексту, а краткое саммари печатаем в штатном формате
echelon3 (``--> ...``) перед каждой валидацией. Модуль без зависимостей от
остального echelon3 — импортируется и из CLI, и из трейнера."""
import warnings
from collections import Counter

_counts: Counter = Counter()
_installed = False


def install():
    """Перехватить показ предупреждений в счётчик (идемпотентно)."""
    global _installed
    if _installed:
        return
    # "always" — чтобы каждый повтор доходил до счётчика (иначе дефолтный дедуп
    # Python показал бы варнинг один раз на место вызова и счёт был бы занижен).
    # Затем префиксом вешаем ignore на заведомый dev-шум — он до счётчика не дойдёт.
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
    """Печатает краткое саммари накопленных предупреждений (tqdm-safe) и очищает."""
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
