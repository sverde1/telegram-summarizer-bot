"""Measured speeds (moving averages) so ETAs match this machine: whisper realtime factor, LLM time."""
import json
import threading

from . import config

_FILE = config.DATA_DIR / "stats.json"
_lock = threading.Lock()


def _load() -> dict:
    try:
        return json.loads(_FILE.read_text())
    except (OSError, ValueError):
        return {}


def get(key: str, default: float) -> float:
    with _lock:
        return float(_load().get(key, default))


def record(key: str, value: float, alpha: float = 0.3) -> None:
    with _lock:
        d = _load()
        d[key] = value if key not in d else (1 - alpha) * d[key] + alpha * value
        _FILE.write_text(json.dumps(d, indent=1))
