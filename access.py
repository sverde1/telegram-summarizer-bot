"""Who may use the bot. Admins come from .env; everyone else lives in the users table."""
import os

from summarizer import config, db  # noqa: F401  (config loads .env)

ADMINS = {int(x) for x in (os.environ.get("ADMIN_USER_IDS") or os.environ.get("ALLOWED_USER_IDS") or "")
          .replace(" ", "").split(",") if x}
STATES = ("allowed", "pending", "blocked")


def is_admin(uid: int) -> bool:
    return uid in ADMINS


def state(uid: int) -> str | None:
    """'admin', 'allowed', 'pending', 'blocked' or None (never seen)."""
    if uid in ADMINS:
        return "admin"
    u = db.get_user(uid)
    return u["status"] if u else None


def set_state(uid: int, new: str | None, name: str | None = None, username: str | None = None) -> dict | None:
    """Move a user to `new` (None = forget). Returns their row, or None if unknown."""
    return db.set_user(uid, new, name, username)


def all_users() -> dict[str, list[dict]]:
    return db.users_by_status()


def label(uid: int, info: dict | None) -> str:
    info = info or {}
    name = info.get("name") or "?"
    return f"{name} (@{info['username']}, {uid})" if info.get("username") else f"{name} ({uid})"
