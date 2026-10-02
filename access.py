"""Who may use the bot. Admins come from .env; everyone else lives in the users table."""
import os
from dataclasses import dataclass

from summarizer import config, db  # noqa: F401  (config loads .env)

# ALLOWED_USER_IDS is the setting's name from before admins existed; still read so old .env files work.
ADMINS = {int(x) for x in (os.environ.get("ADMIN_USER_IDS") or os.environ.get("ALLOWED_USER_IDS") or "")
          .replace(" ", "").split(",") if x}
STATES = ("allowed", "pending", "blocked")  # what admins manage; admins themselves come from .env
# At import: admins need their users row (for their settings) before any handler runs.
db.sync_admins(ADMINS)


def is_admin(uid: int) -> bool:
    """Whether a user is an admin (listed in ADMIN_USER_IDS).

    Args:
        uid: Telegram user id.

    Returns:
        True for admins.
    """
    return uid in ADMINS


def state(uid: int) -> str | None:
    """A user's access state.

    .env is checked first, so an admin stays admin whatever their users row says.

    Args:
        uid: Telegram user id.

    Returns:
        "admin", "allowed", "pending", "blocked", or None if the bot has never seen them.
    """
    if uid in ADMINS:
        return "admin"
    u = db.get_user(uid)
    if not u or u["status"] == "admin":  # an admin row for someone no longer in .env grants nothing
        return None
    return u["status"]


def set_state(uid: int, new: str | None, name: str | None = None, username: str | None = None) -> dict | None:
    """Move a user to another access state, or forget them.

    Removing a user also drops their settings (AI choice).

    Args:
        uid: Telegram user id.
        new: "allowed", "pending" or "blocked"; None = forget the user.
        name: Display name to store; None keeps the stored one.
        username: @username without the @ to store; None keeps the stored one.

    Returns:
        Their row, or None if the user was unknown.
    """
    return db.set_user(uid, new, name, username)


def all_users() -> dict[str, list[dict]]:
    """All users grouped by state.

    Returns:
        {"admin": [...], "allowed": [...], "pending": [...], "blocked": [...]}.
    """
    return db.users_by_status()


def label(uid: int, info: dict | None) -> str:
    """How a user is shown to admins: name, @username and id.

    The id is always included: names aren't unique, and admins need it for ADMIN_USER_IDS.

    Args:
        uid: Telegram user id.
        info: The user's row (or any dict with "name"/"username"), or None.

    Returns:
        E.g. "Ana (@ana, 111)", or "? (111)" when the name isn't known yet.
    """
    info = info or {}
    name = info.get("name") or "?"
    return f"{name} (@{info['username']}, {uid})" if info.get("username") else f"{name} ({uid})"


@dataclass(frozen=True)
class Limit:
    """One per-user daily limit: where its global value and per-user override live.

    Attributes:
        setting: settings-table key of the global value (set with /limit).
        user_column: users-table column of a per-user override (NULL = the global value).
        config_attr: the summarizer.config attribute with the default, read at call time.
    """
    setting: str
    user_column: str
    config_attr: str


# The limits /limit manages. Each also has a usage query (db.USAGE_FILTERS) and a UI entry (bot's LIMIT_UI).
LIMITS = {
    "daily": Limit("daily_limit", "daily_limit", "DAILY_LIMIT"),  # requests (links, files, books)
    "ocr": Limit("ocr_limit", "ocr_limit", "OCR_DAILY_LIMIT"),  # scanned documents read with OCR
    "voice": Limit("tts_limit", "tts_limit", "TTS_DAILY_LIMIT"),  # newly made voice messages
}


def global_limit(kind: str) -> int:
    """A limit for users without an override: the /limit setting, else its config default (0 = no limit)."""
    stored = db.get_setting(LIMITS[kind].setting)
    return int(stored) if stored is not None else getattr(config, LIMITS[kind].config_attr)


def limit(uid: int, kind: str) -> tuple[int | None, bool]:
    """A user's limit of one kind.

    Returns:
        (limit, is_default): limit None for admins (never limited), 0 = no limit; is_default is True when it
        comes from the global setting rather than a per-user override.
    """
    if is_admin(uid):
        return None, False
    u = db.get_user(uid)
    column = LIMITS[kind].user_column
    if u and u.get(column) is not None:
        return u[column], False
    return global_limit(kind), True
