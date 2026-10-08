"""The per-user limits as users see them: usage, refusals and the queue message."""
import time
from dataclasses import dataclass

import access
from summarizer import config, db, stats
from tgbot import render, state


def limit_label(uid: int) -> str:
    """Limit status for the /users list, e.g. "12/100 today (default) · OCR 1/5", or "no limit"."""
    used, limit, is_default, _ = limit_status(uid, "daily")
    if limit is None:  # admins
        return "no limit"
    parts = [f"{used}/{limit} today" + (" (default)" if is_default else "") if limit else "no daily limit"]
    for kind, ui in LIMIT_UI.items():
        if kind != "daily":
            used, limit, _, _ = limit_status(uid, kind)
            parts.append(f"{ui.short} {used}/{limit}" if limit else f"{ui.short} no limit")
    return " · ".join(parts)


@dataclass(frozen=True)
class LimitUI:
    """How a limit (access.LIMITS) is shown.

    Attributes:
        icon: Its emoji in /limit.
        what: What it counts ("requests").
        title: Its name ("Daily limit").
        word: What admins type after /limit to change it ("" for the daily limit, the default).
        short: Its label in the /users list ("OCR").
    """
    icon: str
    what: str
    title: str
    word: str
    short: str


# One entry per access.LIMITS kind, in the order /limit shows them.
LIMIT_UI = {
    "daily": LimitUI("📊", "requests", "Daily limit", "", ""),
    "ocr": LimitUI("🔍", "scanned documents (OCR)", "OCR limit", "ocr", "OCR"),
    "voice": LimitUI("🔊", "new voice messages", "Voice-message limit", "voice", "🔊"),
    "ask": LimitUI("💬", "questions", "Question limit", "ask", "💬"),
}


def usage_line(uid: int, kind: str) -> str:
    """A user's own usage of one limit, e.g. "📊 Today: 12 of your 100 requests (last 24 h). 88 left." """
    ui = LIMIT_UI[kind]
    used, limit, _, frees_in = limit_status(uid, kind)
    icon, what = ui.icon, ui.what
    if not limit:
        return f"{icon} Today: {used} {what} (last 24 h). No limit."
    if used >= limit:
        return (f"{icon} Today: {used} of your {limit} {what} (last 24 h). You can send more in about "
                f"{render.fmt_until(frees_in or 0)}.")
    return f"{icon} Today: {used} of your {limit} {what} (last 24 h). {limit - used} left."


JOB_SECONDS_DEFAULT = 60  # assumed time per job until real jobs have been measured


DAY = 86400  # the daily limit counts links in a rolling 24-hour window


def limit_status(uid: int, kind: str) -> tuple[int, int | None, bool, float | None]:
    """A user's usage of one limit (access.LIMITS) in the rolling 24 hours.

    Returns:
        (used, limit (None for admins, 0 = no limit), whether the limit is the global default, seconds until a
        slot frees up when the limit is reached, else None).
    """
    used, oldest = db.usage(uid, kind, DAY)
    limit, is_default = access.limit(uid, kind)
    frees_in = (oldest + DAY - time.time()) if limit and used >= limit and oldest else None
    return used, limit, is_default, frees_in


def queued_message() -> str:
    """The first reply to a link: "working" if it starts now, else an estimated wait.

    The wait is shown instead of a queue position: a position would tell users how busy the others are.
    It starts right away while a worker is free. Otherwise the jobs ahead (queued, plus the running ones beyond
    the other workers) are shared by config.WORKERS workers; jobs waiting for memory or replaying a cached
    answer don't hold a worker.
    """
    queued, running = state.queue.qsize(), len(state.running)
    if not queued and running < config.WORKERS:
        return "⏳ Got it, working…"
    ahead = queued + running - config.WORKERS + 1
    wait = max(1, ahead) * stats.get("job", JOB_SECONDS_DEFAULT) / config.WORKERS
    return f"⏳ Got it, you're in the queue. Estimated wait: about {render.fmt_eta(wait)}."


def refusal(uid: int, *, new_request: bool = True, count: int = 1) -> str | None:
    """Why a user may not start another job right now, or None if they may (admins always may).

    Args:
        uid: The user.
        new_request: Whether the job creates a request row (counts toward the daily limit); False when it
            continues one (a chapter picked from a list).
        count: How many requests it is (the links of a several-link message); all must fit today's limit.
    """
    if access.is_admin(uid):
        return None
    if state.user_jobs[uid] >= config.MAX_QUEUED_PER_USER:
        return (f"⏳ You already have {state.user_jobs[uid]} requests in the queue. Try again when one of them is "
                "done.")
    if new_request:
        used, limit, _, frees_in = limit_status(uid, "daily")
        if limit and used < limit < used + count:
            return f"⏳ You have {limit - used} requests left today, and these are {count} links. Send fewer."
        if limit and used >= limit:
            return (f"⏳ You've reached today's limit of {limit} requests. You can send more in about "
                    f"{render.fmt_until(frees_in or 0)}.")
    # Every unfinished job counts (queued, running, waiting for memory, replaying), not just the queue.
    # The count isn't shown: it would tell users how busy the others are.
    if len(state.jobs) >= config.MAX_QUEUE:
        return "⏳ The bot is busy right now. Please try again in a few minutes."
    return None


def ocr_refusal(uid: int) -> str | None:
    """The refusal when a non-admin has used up today's OCR, else None."""
    used, limit, _, frees_in = limit_status(uid, "ocr")
    if limit and used >= limit:
        return (f"⏳ This is a scanned document and needs text recognition (OCR). You've used today's {limit} "
                f"OCR documents; you can send more in about {render.fmt_until(frees_in or 0)}.")
    return None
