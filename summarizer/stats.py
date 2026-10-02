"""Measured speeds (moving averages) so ETAs match this machine: whisper realtime factor, LLM time.

Also holds a few small remembered strings (the model each backend's default resolved to, the last
update version the admins were told about). Stored in data/stats.json.
"""
import json
import threading

from . import config

_FILE = config.DATA_DIR / "stats.json"
# The pipeline worker thread, OCR and book threads and the bot's event loop all read and write the stats.
_lock = threading.Lock()
# The file's contents, kept after the first read: ETAs are asked for on every status line. Keyed by the path,
# so pointing _FILE elsewhere (tests) reads that file.
_cache: tuple[object, dict] | None = None


def _load() -> dict:
    """The stored values (read from the file once).

    Returns:
        The stored values, or an empty dict if the file is missing or unreadable (a fresh install,
        or a write interrupted mid-way): ETAs then fall back to their defaults instead of failing.
    """
    global _cache
    if _cache is None or _cache[0] != _FILE:
        try:
            data = json.loads(_FILE.read_text())
        except (OSError, ValueError):
            data = {}
        _cache = (_FILE, data)
    return _cache[1]


def _save(d: dict) -> None:
    """Writes the values, atomically: a crash mid-write can't leave a truncated file."""
    tmp = _FILE.with_name(_FILE.name + ".tmp")
    tmp.write_text(json.dumps(d, indent=1))
    tmp.replace(_FILE)


def get(key: str, default: float) -> float:
    """Returns a measured number, e.g. a whisper realtime factor or an LLM time.

    Args:
        key: Stat name, e.g. "whisper:cpu:small" or "llm:codex".
        default: Value to use before anything has been measured.
    """
    with _lock:
        return float(_load().get(key, default))


def recall(key: str) -> str:
    """Returns a remembered string, or "" if none was stored.

    Args:
        key: E.g. "model:codex" or "update-notified:Codex".
    """
    with _lock:
        return str(_load().get(key, ""))


def remember(key: str, value: str) -> None:
    """Stores a string under `key`, replacing any earlier value."""
    with _lock:
        d = _load()
        d[key] = value
        _save(d)


def record(key: str, value: float, alpha: float = 0.3) -> None:
    """Folds a new measurement into an exponential moving average.

    Args:
        key: Stat name.
        value: The newly measured value.
        alpha: Weight of the new value. 0.3 lets ETAs follow real changes (another model, a GPU)
            within a few jobs while one unusually slow or fast job doesn't swing them much.
    """
    with _lock:
        d = _load()
        # The first measurement is taken as is: averaging it with a guessed default would only blur it.
        d[key] = value if key not in d else (1 - alpha) * d[key] + alpha * value
        _save(d)
