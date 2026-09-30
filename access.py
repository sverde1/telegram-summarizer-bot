"""Who may use the bot. Admins come from .env; everyone else lives in data/users.json.

{"allowed": {id: info}, "pending": {id: info}, "blocked": {id: info}}  (JSON keys are strings)
info = {"name": ..., "username": ..., "at": unix time}
"""
import json
import os
import threading
import time

from summarizer import config

_FILE = config.DATA_DIR / "users.json"
_lock = threading.Lock()
ADMINS = {int(x) for x in (os.environ.get("ADMIN_USER_IDS") or os.environ.get("ALLOWED_USER_IDS") or "")
          .replace(" ", "").split(",") if x}
STATES = ("allowed", "pending", "blocked")


def _load() -> dict:
    try:
        d = json.loads(_FILE.read_text())
    except (OSError, ValueError):
        d = {}
    return {s: d.get(s, {}) for s in STATES}


def _save(d: dict) -> None:
    tmp = _FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1, ensure_ascii=False))
    tmp.replace(_FILE)


def is_admin(uid: int) -> bool:
    return uid in ADMINS


def state(uid: int) -> str | None:
    """'admin', 'allowed', 'pending', 'blocked' or None (never seen)."""
    if uid in ADMINS:
        return "admin"
    d = _load()
    return next((s for s in STATES if str(uid) in d[s]), None)


def set_state(uid: int, new: str | None, info: dict | None = None) -> dict | None:
    """Move a user to `new` (None = forget). Returns the user's info, or None if unknown."""
    with _lock:
        d = _load()
        old = None
        for s in STATES:
            old = d[s].pop(str(uid), None) or old
        info = {**(old or {}), **(info or {}), "at": time.time()}
        if new:
            d[new][str(uid)] = info
        _save(d)
    return info if (old or new) else None


def all_users() -> dict:
    return _load()


def label(uid: int | str, info: dict) -> str:
    name = info.get("name") or "?"
    return f"{name} (@{info['username']}, {uid})" if info.get("username") else f"{name} ({uid})"
