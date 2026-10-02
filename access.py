"""Who may use the bot. Admins come from .env; everyone else lives in the users table."""
import os

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


def global_daily_limit() -> int:
    """Links per 24 h for users without an override: the /limit setting, else DAILY_LIMIT (0 = no limit)."""
    stored = db.get_setting("daily_limit")
    return int(stored) if stored is not None else config.DAILY_LIMIT


def daily_limit(uid: int) -> tuple[int | None, bool]:
    """A user's daily limit.

    Returns:
        (limit, is_default): limit None for admins (never limited), 0 = no limit; is_default is True when it
        comes from the global setting rather than a per-user override.
    """
    if is_admin(uid):
        return None, False
    u = db.get_user(uid)
    if u and u.get("daily_limit") is not None:
        return u["daily_limit"], False
    return global_daily_limit(), True


def global_ocr_limit() -> int:
    """OCR runs per 24 h for users without an override: the /limit ocr setting, else OCR_DAILY_LIMIT (0 = none)."""
    stored = db.get_setting("ocr_limit")
    return int(stored) if stored is not None else config.OCR_DAILY_LIMIT


def ocr_limit(uid: int) -> tuple[int | None, bool]:
    """A user's OCR limit, like daily_limit: (limit or None for admins, whether it's the global default)."""
    if is_admin(uid):
        return None, False
    u = db.get_user(uid)
    if u and u.get("ocr_limit") is not None:
        return u["ocr_limit"], False
    return global_ocr_limit(), True
