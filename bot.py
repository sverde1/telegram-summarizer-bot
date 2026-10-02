"""Telegram bot: send a YouTube/TikTok link, get back title, clickbait answer and summary."""
import asyncio
import collections
import math
import datetime as dt
import html
import io
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

# Opt in to PTB's coming behavior: RetryAfter.retry_after as a timedelta (an int with a deprecation warning
# until then). Must be set before telegram is imported; _retry_seconds handles both forms.
os.environ.setdefault("PTB_TIMEDELTA", "1")

from telegram import (BotCommand, BotCommandScopeChat, BotCommandScopeDefault, InlineKeyboardButton,
                      InlineKeyboardMarkup, Update)
from telegram.constants import ParseMode
from telegram.error import BadRequest, Forbidden, RetryAfter, TelegramError
from telegram.ext import (Application, CallbackQueryHandler, ChatMemberHandler, CommandHandler, ContextTypes,
                          MessageHandler, filters)

import access
from summarizer import (config, db, documents, links, memory, ocr, pipeline, proc, stats, summarize, transcribe,
                        updates)
from summarizer.urls import UnsupportedURL, check as check_url, find_url

logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO)
# httpx logs every Telegram API call (each long poll, each status edit) at INFO; that drowns the bot's log.
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("bot")

TG_LIMIT = 4096  # Telegram's maximum message length in characters
HELP = (
    "Send me a YouTube or TikTok link and I'll reply with the title, an answer to any clickbait, "
    "and a summary.\n\n"
    "/again <url> - ignore the cache and summarize again\n"
    "/transcript <url> - send the raw transcript as a file\n"
    "/history - your recent requests\n"
    "/models - show or choose the AI (Codex or Claude) and model\n"
    "/limit - how many requests you can still send today\n\n"
    "📄 You can also send a book or document (PDF, EPUB, DOCX or TXT, up to 20 MB), or a Google Drive or "
    "Dropbox link to one (shared as \"Anyone with the link\"): I'll summarize the whole thing or chapter by "
    "chapter."
)
ADMIN_HELP = ("\n\nAdmin:\n/users - list users; allow, remove, or unblock them\n"
              "/history - recent requests from all users (who sent what, cache hits)\n"
              "/limit - daily limits: /limit 50 (everyone), /limit <user id> 200 (one user), "
              "/limit <user id> default, /limit 0 (no limit); /limit ocr … for scanned documents\n"
              "/ocrlang - OCR languages: /ocrlang add slv, /ocrlang remove slv")


@dataclass
class Job:
    """One queued request: what to process, where to reply, and on whose behalf.

    Attributes:
        url: The link the user sent.
        chat_id: Chat to reply in.
        status_id: Message id of the status message that gets edited while the job runs.
        use_cache: False for /again (re-summarize, ignoring the cached summary).
        transcript_only: True for /transcript (send the transcript, no summary).
        queued_at: time.monotonic() when the job was queued, to report time spent waiting.
        user_id: Telegram user id of the requester.
        request_id: Row id in the `requests` table, updated when the job finishes.
        backend: The user's chosen LLM backend; None = the default.
        model: The user's chosen model of that backend; None = the default.
    """

    url: str
    chat_id: int
    status_id: int
    use_cache: bool = True
    transcript_only: bool = False
    queued_at: float = 0.0
    user_id: int = 0
    request_id: int = 0
    backend: str | None = None  # the user's chosen LLM backend/model; None = default
    model: str | None = None
    cancel_reason: str | None = None  # set when the job is cancelled; the text shown to the user
    waiting_since: float | None = None  # when it was first set aside for lack of memory (monotonic)
    memory_needed: int = 0  # bytes its transcription needs (shown while it waits)
    audio_seconds: float = 0  # its audio length, to re-estimate the memory need on each re-check
    upload_id: int = 0  # an uploaded document (url is then its file name); 0 for links
    book_mode: str = ""  # whole | short | each | pick
    chapter: int | None = None  # for "pick": the chosen chapter (None: show the chapter list)
    ocr_ok: bool = False  # the user confirmed OCR of this scanned document


# A single queue drained by a single worker: jobs run one at a time, so Whisper (CPU-heavy) never runs
# twice in parallel, and two users' LLM conversations can't interleave.
queue: asyncio.Queue[Job] = asyncio.Queue()


ACCESS_REMOVED = "⛔ Your access to this bot was removed, so this request was cancelled."

# Every job not yet finished, by request id, so a user's jobs can be found and cancelled. The worker runs one
# job at a time; _running is that one (its programs are stopped through proc.current_job_cancel).
_jobs: dict[int, "Job"] = {}
_running: "Job | None" = None


async def _report_cancel(app: Application, job: "Job") -> None:
    """Records a job as cancelled and shows the reason on its status message."""
    reason = job.cancel_reason or CANCELLED
    db.update_request(job.request_id, status="cancelled", error="cancelled: " + reason[:200])
    await _fail(app, job, reason)


async def _cancel_job(app: Application, job: "Job", reason: str) -> bool:
    """Cancels one job, whatever state it's in, and tells its user why.

    The running job only gets its programs killed (via the cancel event); the worker then reports it. Every
    other state is finished right here (status message, request row, registry, the user's slot): a queued job
    is later skipped by the worker, a parked one leaves the memory-wait list, and a replay task is cancelled,
    possibly before it ever ran, so its own cleanup can't be relied on.

    Args:
        app: The running application.
        job: The job to cancel.
        reason: The message shown on its status.

    Returns:
        False if the job was already cancelled or finished.
    """
    if job.cancel_reason or job.request_id not in _jobs:
        return False
    job.cancel_reason = reason
    if job is _running:
        proc.current_job_cancel.set()
        return True
    task = _delayed.pop(job.request_id, None)
    if task:
        task.cancel()
    _unpark(job)
    _end_job(job)
    await _report_cancel(app, job)
    return True


async def cancel_user_jobs(app: Application, uid: int, reason: str) -> int:
    """Cancels all of a user's queued and running jobs and tells them why.

    Queued jobs are marked (the worker skips them) and their status message changes at once; the running job
    has its current program killed and stops at the next checkpoint.

    Args:
        app: The running application.
        uid: The user whose jobs to cancel.
        reason: The message shown on each cancelled job's status.

    Returns:
        How many jobs were cancelled.
    """
    mine = [j for j in _jobs.values() if j.user_id == uid and not j.cancel_reason]
    for job in mine:
        await _cancel_job(app, job, reason)
    if mine:
        log.info("cancelled %d job(s) of user %s", len(mine), uid)
    return len(mine)


# Jobs per user, queued or running (see config.MAX_QUEUED_PER_USER). Decremented whenever a job ends,
# however it ends, so a user is never locked out by a job that's gone.
_user_jobs: collections.Counter = collections.Counter()

MAX_PENDING = 10  # open access requests; more is a flood of throwaway accounts, not family and friends
PENDING_REPLY_EVERY = 600  # seconds between "still waiting for approval" replies to the same user
_pending_replied: dict[int, float] = {}  # user id -> when they last got that reply


async def guard(update: Update, ctx: ContextTypes.DEFAULT_TYPE, request: bool = False) -> bool:
    """Checks whether the user may use the bot, and handles everyone who may not.

    Strangers only get an access request sent to the admins when they explicitly ask with /start, so
    random messages to the bot never ping the admins. Blocked users are ignored silently.

    Args:
        update: The incoming update.
        ctx: Handler context (used to message the admins).
        request: True for /start: file an access request for an unknown user.

    Returns:
        True if the user is an admin or allowed; False otherwise (they've been told why, if appropriate).
    """
    user = update.effective_user
    if not user:
        return False
    st = access.state(user.id)
    if st in ("admin", "allowed"):
        return True
    msg = update.effective_message
    if not access.ADMINS:  # setup mode: tell the owner their id
        if msg:
            await msg.reply_text(f"Setup: your Telegram user id is {user.id}. Add ADMIN_USER_IDS={user.id} "
                                 "to .env and restart the bot.")
        return False
    # Below, everyone who can't use the bot is answered sparingly: each reply costs a Telegram call on the
    # single update-processing path, so a spammer would otherwise slow the bot down for everyone.
    if st == "blocked":
        return False  # silently, and without a log line per message
    if st == "pending":
        # Already asked (admins were notified once); remind them at most every PENDING_REPLY_EVERY seconds.
        now = time.monotonic()
        if msg and now - _pending_replied.get(user.id, -PENDING_REPLY_EVERY) >= PENDING_REPLY_EVERY:
            _pending_replied[user.id] = now
            await msg.reply_text("⏳ Your access request is waiting for the admin's approval.")
        return False
    if not request:
        return False  # strangers only get an answer to /start, the one command their menu shows
    if len(access.all_users()["pending"]) >= MAX_PENDING:
        log.warning("access request from %s refused: %d requests already pending", user.id, MAX_PENDING)
        if msg:
            await msg.reply_text("🔒 This bot isn't accepting new access requests right now. Please try later.")
        return False
    info = access.set_state(user.id, "pending", user.full_name, user.username)
    log.warning("access request from %s", access.label(user.id, info))
    buttons = InlineKeyboardMarkup([[InlineKeyboardButton("✅ Allow", callback_data=f"allow:{user.id}"),
                                     InlineKeyboardButton("❌ Deny", callback_data=f"block:{user.id}")]])
    for admin in access.ADMINS:
        try:
            await ctx.bot.send_message(admin, f"🔔 Access request from {access.label(user.id, info)}",
                                       reply_markup=buttons)
        except (BadRequest, Forbidden) as e:
            log.error("couldn't notify admin %s: %s", admin, e)
    if msg:
        await msg.reply_text("🔒 This is a private bot. I've asked the admin to give you access; "
                             "you'll get a message when it's approved.")
    return False


async def on_users(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles /users (admins only): lists admins and users, with action buttons per user.

    Sends one message for the admins, then one per non-empty group (allowed, pending, blocked) so each
    group gets its own buttons. Non-admins get no reply at all, so the command's existence isn't revealed.
    """
    user = update.effective_user
    if not access.is_admin(user.id):
        return
    db.touch_user(user.id, user.full_name, user.username)  # admins have no /start request to record it
    users = access.all_users()

    def llm(u: dict) -> str:
        """Describes a user's AI choice, e.g. "Codex · gpt-6-sol" or "default AI".

        Args:
            u: A users-table row.

        Returns:
            Short text for the user list.
        """
        if not u.get("backend") and not u.get("model"):
            return "default AI"
        return f"{summarize.BACKEND_NAMES.get(u['backend'], u['backend'] or '')} · {u['model'] or 'default'}"

    admins = "\n".join(f"• {access.label(u['id'], u)}: {llm(u)} · no limit" for u in users["admin"])
    # Admins have a users row too (status "admin", for their settings) but aren't one of the managed
    # STATES; count only the managed ones, or the "no other users" hint would never show.
    others = sum(len(users[st]) for st in access.STATES)
    await update.message.reply_text(
        f"👑 Admins (set in .env)\n{admins}"
        + ("" if others else "\n\nNo other users yet. When someone sends the bot /start, "
                             "you'll get an access request here."))
    # "remove" doubles as Unblock: forgetting a blocked user lets them /start a new request.
    actions = {"allowed": [("🗑 Remove", "remove")], "pending": [("✅ Allow", "allow"), ("❌ Deny", "block")],
               "blocked": [("↩️ Unblock", "remove")]}
    glob = access.global_daily_limit()
    titles = {"allowed": f"✅ Allowed (daily limit: {glob or 'none'})", "pending": "⏳ Pending",
              "blocked": "⛔ Blocked"}
    for st in access.STATES:
        if not users[st]:
            continue
        rows = [[InlineKeyboardButton(f"{text}: {u['name'] or u['id']}", callback_data=f"{act}:{u['id']}")
                 for text, act in actions[st]] for u in users[st]]
        lines = "\n".join(f"• {access.label(u['id'], u)}"
                          + (f": {llm(u)} · {_limit_label(u['id'])}" if st == "allowed" else "")
                          for u in users[st])
        await update.message.reply_text(f"{titles[st]}\n{lines}", reply_markup=InlineKeyboardMarkup(rows))


def _current_llm(uid: int) -> tuple[str, str, bool]:
    """Resolves which backend and model a user's summaries use.

    Args:
        uid: Telegram user id.

    Returns:
        (backend, model, is_default): the effective backend and model, and whether the user is on the
        defaults (made no choice, or their chosen backend is no longer installed).
    """
    backend, model = db.get_user_llm(uid)
    if backend not in summarize.available_backends():
        backend, model = None, None  # their choice was uninstalled: fall back
    b = backend or config.LLM_BACKEND
    return b, model or summarize.default_model(b), not backend and not model


def _llm_home(uid: int) -> tuple[str, InlineKeyboardMarkup]:
    """Builds step 1 of /models: the user's current AI and a button per installed provider.

    Args:
        uid: Telegram user id.

    Returns:
        Message text and its inline keyboard.
    """
    b, m, is_default = _current_llm(uid)
    text = (f"🧠 You're using {summarize.BACKEND_NAMES[b]} · {m}" + (" (default)" if is_default else "")
            + "\n\nChoose a provider:")
    rows = []
    for backend in summarize.available_backends():
        name, billing = summarize.BACKENDS[backend]
        mark = "✓ " if backend == b else ""
        tag = " (default)" if backend == config.LLM_BACKEND else ""
        rows.append([InlineKeyboardButton(f"{mark}{name} ({billing}){tag}", callback_data=f"llm:b:{backend}")])
    if not is_default:
        rows.append([InlineKeyboardButton("↩️ Back to default", callback_data="llm:default")])
    return text, InlineKeyboardMarkup(rows)


async def _llm_models(uid: int, backend: str) -> tuple[str, InlineKeyboardMarkup]:
    """Builds step 2 of /models: the models of one provider, the user's current one marked ✓.

    Args:
        uid: Telegram user id.
        backend: The provider whose models to list.

    Returns:
        Message text and its inline keyboard (an error text with only a Back button if listing fails).
    """
    b, m, _ = _current_llm(uid)
    try:
        # The API backends fetch their list over the network: never on the event loop, which would
        # freeze every other chat until the provider answers.
        models = await asyncio.to_thread(summarize.list_models, backend)
    except summarize.SummaryError as e:
        return f"⚠️ {e}", InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data="llm:home")]])
    default = summarize.default_model(backend)
    lines = [f"🧠 {summarize.BACKEND_NAMES[backend]} models:", ""]
    rows = []
    for model in models:
        mark = "✓ " if (backend, model["id"]) == (b, m) else ""
        tag = " (default)" if model["id"] == default else ""
        lines.append(f"{mark}{model['id']}{tag}" + (f": {model['description']}" if model["description"] else ""))
        # Telegram rejects callback_data longer than 64 bytes.
        rows.append([InlineKeyboardButton(f"{mark}{model['name'] or model['id']}{tag}",
                                          callback_data=f"llm:m:{backend}:{model['id']}"[:64])])
    rows.append([InlineKeyboardButton("⬅️ Back", callback_data="llm:home")])
    return "\n".join(lines), InlineKeyboardMarkup(rows)


async def _set_llm(uid: int, backend: str | None, model: str | None) -> str:
    """Validates and stores a user's backend/model choice.

    The choice is re-validated here rather than trusted from the button, because callback data comes
    from the client and the model list can change between showing the buttons and the tap.

    Args:
        uid: Telegram user id.
        backend: The chosen backend, or None to go back to the defaults.
        model: The chosen model of that backend.

    Returns:
        The confirmation (or error) text to show the user.
    """
    if backend is None:
        db.set_user_llm(uid, None, None)
        b = config.LLM_BACKEND
        return f"✅ Back to the default: {summarize.BACKEND_NAMES[b]} · {summarize.default_model(b)}"
    if backend not in summarize.available_backends():
        return f"⚠️ {backend} isn't available on this bot."
    try:
        ids = [m["id"] for m in await asyncio.to_thread(summarize.list_models, backend)]
    except summarize.SummaryError as e:
        return f"⚠️ {e}"
    if model not in ids:
        return f"Unknown {summarize.BACKEND_NAMES[backend]} model “{model}”. Available: {', '.join(ids)}"
    db.set_user_llm(uid, backend, model)
    return (f"✅ Your summaries now use {summarize.BACKEND_NAMES[backend]} · {model}. "
            "Each video gets a summary written by this model (kept separately from other models' summaries).")


async def on_models(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles /models: shows step 1 of the provider/model picker."""
    if await guard(update, ctx):
        text, buttons = _llm_home(update.effective_user.id)
        await update.message.reply_text(text, reply_markup=buttons)


def _private(update: Update) -> bool:
    """Whether the update comes from a private chat with the bot.

    The bot is for private chats only: in a group, summaries, /history and /users would be shown to every
    member. Message handlers filter on this; button handlers call it (buttons have no chat-type filter).
    """
    return bool(update.effective_chat and update.effective_chat.type == "private")


async def on_my_chat_member(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Leaves any group or channel the bot is added to, and tells the admins who added it.

    BotFather's "Allow Groups" setting should prevent this (see README); this is the fallback if it's on.
    """
    change = update.my_chat_member
    chat = change.chat
    if chat.type == "private" or change.new_chat_member.status in ("left", "kicked"):
        return  # private chats are normal, and leaving needs no reaction
    who = change.from_user
    try:
        await ctx.bot.leave_chat(chat.id)
    except TelegramError as e:
        log.error("couldn't leave chat %s: %s", chat.id, e)
    log.warning("added to %s %r by %s; left", chat.type, chat.title, who and who.id)
    text = (f"⚠️ {access.label(who.id, {'name': who.full_name, 'username': who.username}) if who else 'Someone'} "
            f"added the bot to the {chat.type} “{chat.title or chat.id}”. I left it: the bot only works in "
            "private chats.")
    for admin in access.ADMINS:
        try:
            await ctx.bot.send_message(admin, text)
        except TelegramError as e:
            log.warning("couldn't notify admin %s: %s", admin, e)


async def on_llm_button(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles taps in the /models picker by editing the same message in place.

    Callback data: `llm:home`, `llm:default`, `llm:b:<backend>` (open a provider's models) or
    `llm:m:<backend>:<model>` (choose a model).
    """
    q = update.callback_query
    uid = q.from_user.id
    # Buttons can outlive access (e.g. the user was removed after /models was shown): re-check on every tap.
    if not _private(update) or access.state(uid) not in ("admin", "allowed"):
        await q.answer()
        return
    parts = (q.data or "").split(":", 3)  # llm:home | llm:default | llm:b:<backend> | llm:m:<backend>:<model>
    # Callback data comes from the client: anything malformed or naming an unknown provider just shows the
    # first step again instead of raising.
    action = parts[1] if len(parts) > 1 else ""
    if action == "b" and len(parts) == 3 and parts[2] in summarize.BACKENDS:
        text, buttons = await _llm_models(uid, parts[2])
        await q.answer()
    elif action == "m" and len(parts) == 4 and parts[2] in summarize.BACKENDS:
        reply = await _set_llm(uid, parts[2], parts[3])
        await q.answer(reply[:200])  # Telegram caps callback answer (toast) text at 200 characters
        text, buttons = await _llm_models(uid, parts[2])
    elif action == "default":
        reply = await _set_llm(uid, None, None)
        await q.answer(reply[:200])
        text, buttons = _llm_home(uid)
    else:
        text, buttons = _llm_home(uid)
        await q.answer()
    try:
        await q.edit_message_text(text, reply_markup=buttons)
    except BadRequest:  # unchanged
        pass


def _limit_label(uid: int) -> str:
    """Limit status for the /users list, e.g. "12/100 today (default) · OCR 1/5", or "no limit"."""
    used, limit, is_default, _ = _daily_status(uid)
    if limit is None:  # admins
        return "no limit"
    daily = f"{used}/{limit} today" + (" (default)" if is_default else "") if limit else "no daily limit"
    used, limit, _, _ = _ocr_status(uid)
    return f"{daily} · OCR {used}/{limit}" if limit else f"{daily} · OCR no limit"


# The two limits /limit manages: setting key, per-user setter, global getter, status, what is counted.
LIMITS = {
    "daily": ("daily_limit", db.set_user_daily_limit, access.global_daily_limit, lambda uid: _daily_status(uid),
              "requests", "Daily limit"),
    "ocr": ("ocr_limit", db.set_user_ocr_limit, access.global_ocr_limit, lambda uid: _ocr_status(uid),
            "scanned documents (OCR)", "OCR limit"),
}


def _usage_line(uid: int, kind: str) -> str:
    """A user's own usage of one limit, e.g. "📊 Today: 12 of your 100 requests (last 24 h). 88 left." """
    _, _, _, status, what, _ = LIMITS[kind]
    used, limit, _, frees_in = status(uid)
    icon = "📊" if kind == "daily" else "🔍"
    if not limit:
        return f"{icon} Today: {used} {what} (last 24 h). No limit."
    if used >= limit:
        return (f"{icon} Today: {used} of your {limit} {what} (last 24 h). You can send more in about "
                f"{_fmt_until(frees_in or 0)}.")
    return f"{icon} Today: {used} of your {limit} {what} (last 24 h). {limit - used} left."


async def on_limit(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles /limit: users see their own usage; admins see and change the limits.

    Admin forms, each also with "ocr" first for the OCR limit (e.g. `/limit ocr 5`): `/limit` (show),
    `/limit 50` (everyone), `/limit <user id> 200` (one user), `/limit <user id> default` (remove the
    override), `0` meaning no limit.
    """
    if not await guard(update, ctx):
        return
    uid = update.effective_user.id
    if not access.is_admin(uid):
        await update.message.reply_text(f"{_usage_line(uid, 'daily')}\n{_usage_line(uid, 'ocr')}")
        return
    args = list(ctx.args)
    kind = "ocr" if args and args[0].lower() == "ocr" else "daily"
    if kind == "ocr":
        args = args[1:]
    key, set_user, glob, _, what, title = LIMITS[kind]
    if not args:
        lines = []
        for k, (_, _, g, _, w, t) in LIMITS.items():
            lines.append(f"📊 {t} for everyone: {g() or 'none'} {w} per 24 h.")
            col = "daily_limit" if k == "daily" else "ocr_limit"
            lines += [f"  • {access.label(u['id'], u)}: {u[col] or 'no limit'}"
                      for u in access.all_users()["allowed"] if u.get(col) is not None]
        lines.append("\nChange it: /limit 50 · one user: /limit <user id> 200 · /limit <user id> default · "
                     "0 = no limit. The same with \"ocr\" first for the OCR limit: /limit ocr 5")
        await update.message.reply_text("\n".join(lines))
        return
    if len(args) == 1 and args[0].isdigit():
        db.set_setting(key, str(int(args[0])))
        await update.message.reply_text(f"✅ {title} for everyone: {int(args[0]) or 'none'}.")
        return
    if len(args) == 2 and args[0].isdigit() and (args[1].isdigit() or args[1] == "default"):
        target = int(args[0])
        value = None if args[1] == "default" else int(args[1])
        if access.is_admin(target) or not set_user(target, value):
            await update.message.reply_text("⚠️ No such user (admins have no limit).")
            return
        who = access.label(target, db.get_user(target))
        text = "back to the default" if value is None else (value or "no limit")
        await update.message.reply_text(f"✅ {title} for {who}: {text}.")
        return
    await update.message.reply_text("Usage: /limit · /limit 50 · /limit <user id> 200 · /limit <user id> default "
                                    "(add \"ocr\" first for the OCR limit)")


async def on_ocrlang(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles /ocrlang (admins only): lists, adds or removes the languages OCR can read.

    `/ocrlang` lists the installed and available languages; `/ocrlang add <code>` downloads the model for
    the current engine (from fixed sources only) and adds it; `/ocrlang remove <code>` drops one.
    Non-admins get no reply, so the command's existence isn't revealed.
    """
    uid = update.effective_user.id
    if not access.is_admin(uid):
        return
    args = [a.lower() for a in ctx.args]
    if len(args) == 2 and args[0] in ("add", "remove"):
        if args[0] == "add" and args[1] in ocr.LANGUAGES and args[1] not in ocr.installed():
            await update.message.reply_text(f"⏳ Downloading the {ocr.LANGUAGES[args[1]][0]} model…")
        try:
            fn = ocr.add_language if args[0] == "add" else ocr.remove_language
            reply = await asyncio.to_thread(fn, args[1])  # a download takes a while: off the event loop
        except ocr.LanguageError as e:
            reply = f"⚠️ {e}"
        await update.message.reply_text(reply)
        return
    have = ocr.installed()
    others = ", ".join(f"{code} {name}" for code, (name, *_rest) in ocr.LANGUAGES.items() if code not in have)
    engine = {"tesseract": "Tesseract", "rapidocr": "RapidOCR"}[ocr.engine()]
    status = "" if ocr.available() else " (not installed!)"
    await update.message.reply_text(
        f"🔍 OCR engine: {engine}{status}\nInstalled: " + ", ".join(f"{ocr.LANGUAGES[c][0]} ({c})" for c in have)
        + f"\n\nAdd: /ocrlang add <code> · remove: /ocrlang remove <code>\nAvailable: {others}")


async def on_history(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles /history: recent requests, admins see everyone's (with who sent them), users only their own.

    Who submitted what is private, so only admins get other users' rows and the ⚡ cache-hit marker.
    """
    if not await guard(update, ctx):
        return
    uid = update.effective_user.id
    admin = access.is_admin(uid)
    rows = db.recent_requests(None if admin else uid, limit=20)
    if not rows:
        await update.message.reply_text("No requests yet.")
        return
    icons = {"done": "✅", "failed": "⚠️", "queued": "⏳", "processing": "⏳", "cancelled": "✖️", "waiting": "⏸"}
    lines = []
    for r in rows:
        when = time.strftime("%d.%m. %H:%M", time.localtime(r["created_at"]))
        line = f"{icons.get(r['status'], '•')} {when} "
        if admin:
            who = "you" if r["user_id"] == uid else (r["user_name"] or str(r["user_id"]))
            line += f"[{who}] "
        line += (r["title"] or r["url"])[:70]
        if r["kind"] != "summary":
            line += f" ({r['kind']})"
        if admin and r["cached"]:
            line += " ⚡"
        lines.append(line)
    head = "Recent requests (all users; ⚡ = from cache)" if admin else "Your recent requests"
    await update.message.reply_text(head + "\n\n" + "\n".join(lines), disable_web_page_preview=True)


async def on_cancel_button(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles "✖️ Don't wait, cancel" on a job waiting for memory (`cancel:<request id>`).

    Only the job's owner or an admin may cancel it; the data is checked, since callback data can be forged.
    """
    q = update.callback_query
    _, _, rid = (q.data or "").partition(":")
    job = _jobs.get(int(rid)) if rid.isdigit() else None
    if not _private(update) or job is None or job.cancel_reason:
        await q.answer("This request isn't waiting any more.")
        return
    if q.from_user.id != job.user_id and not access.is_admin(q.from_user.id):
        await q.answer("Only the person who sent this link can cancel it.")
        return
    await _cancel_job(ctx.application, job, CANCELLED)
    await q.answer("Cancelled.")


async def on_button(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the admins' user-management buttons (`allow:<id>`, `block:<id>`, `remove:<id>`).

    Used by both the access-request message and /users. After the change the user's command menu is
    re-synced, and a newly allowed user is told they're in.
    """
    q = update.callback_query
    # Only the admins get these buttons, but callback data can be forged: check on every tap.
    if not _private(update) or not access.is_admin(q.from_user.id):
        await q.answer("Admins only.")
        return
    action, _, uid_s = (q.data or "").partition(":")
    if action not in ("allow", "block", "remove") or not uid_s.isdigit():
        await q.answer()
        return
    uid = int(uid_s)
    # Admins come from .env; changing them here would be undone at the next restart anyway.
    if access.is_admin(uid):
        await q.answer("That's an admin (configured in .env).")
        return
    new = {"allow": "allowed", "block": "blocked", "remove": None}[action]
    info = access.set_state(uid, new) or {}
    who = access.label(uid, info)
    done = {"allow": f"✅ Allowed {who}", "block": f"⛔ Denied and blocked {who}",
            "remove": f"🗑 Removed {who}"}[action]
    log.info("admin %s: %s", q.from_user.id, done)
    if action in ("block", "remove"):
        await cancel_user_jobs(ctx.application, uid, ACCESS_REMOVED)
    await q.answer(done[:200])  # Telegram caps callback answer (toast) text at 200 characters
    await q.edit_message_text(done)
    await sync_commands(ctx.bot, uid)
    if action == "allow":
        try:
            await ctx.bot.send_message(uid, "✅ You now have access. Send me a YouTube or TikTok link.\n\n" + HELP)
        except (BadRequest, Forbidden) as e:
            log.warning("couldn't notify user %s: %s", uid, e)


JOB_SECONDS_DEFAULT = 60  # assumed time per job until real jobs have been measured
DAY = 86400  # the daily limit counts links in a rolling 24-hour window


def _fmt_until(seconds: float) -> str:
    """Coarse time until something, rounded up: "25 min" under an hour, else "3 h"."""
    if seconds < 3600:
        return f"{max(1, math.ceil(seconds / 60))} min"
    return f"{math.ceil(seconds / 3600)} h"


def _daily_status(uid: int) -> tuple[int, int | None, bool, float | None]:
    """A user's usage against their daily limit.

    Returns:
        (used in the last 24 h, limit (None for admins, 0 = no limit), whether the limit is the global
        default, seconds until a slot frees up when the limit is reached, else None).
    """
    used, oldest = db.daily_usage(uid, DAY)
    limit, is_default = access.daily_limit(uid)
    frees_in = (oldest + DAY - time.time()) if limit and used >= limit and oldest else None
    return used, limit, is_default, frees_in


def _queued_message() -> str:
    """The first reply to a link: "working" if it starts now, else an estimated wait.

    The wait is shown instead of a queue position: a position would tell users how busy the others are.
    Jobs ahead are the queued ones plus the running one; jobs waiting for memory or replaying a cached
    answer don't hold the queue up.
    """
    ahead = queue.qsize() + (1 if _running is not None else 0)
    if not ahead:
        return "⏳ Got it, working…"
    wait = ahead * stats.get("job", JOB_SECONDS_DEFAULT)
    return f"⏳ Got it, you're in the queue. Estimated wait: about {_fmt_eta(wait)}."


def _refusal(uid: int, *, new_request: bool = True) -> str | None:
    """Why a user may not start another job right now, or None if they may (admins always may).

    Args:
        uid: The user.
        new_request: Whether the job creates a request row (counts toward the daily limit); False when it
            continues one (a chapter picked from a list).
    """
    if access.is_admin(uid):
        return None
    if _user_jobs[uid] >= config.MAX_QUEUED_PER_USER:
        return (f"⏳ You already have {_user_jobs[uid]} requests in the queue. Try again when one of them is "
                "done.")
    if new_request:
        used, limit, _, frees_in = _daily_status(uid)
        if limit and used >= limit:
            return (f"⏳ You've reached today's limit of {limit} requests. You can send more in about "
                    f"{_fmt_until(frees_in or 0)}.")
    # Every unfinished job counts (queued, running, waiting for memory, replaying), not just the queue.
    # The count isn't shown: it would tell users how busy the others are.
    if len(_jobs) >= config.MAX_QUEUE:
        return "⏳ The bot is busy right now. Please try again in a few minutes."
    return None


async def _start_job(uid: int, chat_id: int, status_id: int, url: str, kind: str, request_id: int | None = None,
                     **opts) -> Job:
    """Logs the request (unless it continues one) and queues the job; the caller sent the status message.

    Args:
        uid: The requesting user.
        chat_id: Their chat.
        status_id: The status message the worker edits as the job progresses.
        url: The link, or the file name for documents.
        kind: The request kind (summary, again, transcript, book, ...).
        request_id: An existing request to continue, or None for a new one.
        **opts: Job fields (use_cache, transcript_only, upload_id, book_mode, chapter).
    """
    req = request_id or db.add_request(uid, url, kind)  # logged the moment it arrives
    b, m, is_default = _current_llm(uid)
    # Users on the defaults pass None, so their jobs follow the default (and its later changes)
    # instead of pinning whatever the default resolves to right now.
    b, m = (None, None) if is_default else (b, m)
    _user_jobs[uid] += 1
    job = Job(url, chat_id, status_id, queued_at=time.monotonic(), user_id=uid, request_id=req, backend=b,
              model=m, **opts)
    _jobs[req] = job
    await queue.put(job)
    return job


async def enqueue(update: Update, url: str | None, **opts) -> None:
    """Acknowledges a link right away, logs the request, and queues the job.

    The status message is sent before queueing so the user gets an answer immediately; the worker then
    edits that same message as the job progresses.

    Args:
        update: The incoming update (an allowed user's message or command).
        url: The link found in the message, or None.
        **opts: Job options: `use_cache=False` for /again, `transcript_only=True` for /transcript.
    """
    if not url:
        await update.message.reply_text("Send me a YouTube or TikTok link, or a book or document.")
        return
    try:
        check_url(url)  # no network: a bad link is refused before it gets a queue slot or a request row
    except UnsupportedURL as e:
        await update.message.reply_text(f"⚠️ {e}")
        return
    uid = update.effective_user.id
    if refusal := _refusal(uid):
        await update.message.reply_text(refusal)
        return
    status = await update.message.reply_text(_queued_message())
    kind = "transcript" if opts.get("transcript_only") else "again" if opts.get("use_cache") is False else "summary"
    await _start_job(uid, update.effective_chat.id, status.message_id, url, kind, **opts)


# ---------- uploaded documents ----------

TG_DOWNLOAD_LIMIT = 20 * 1024 ** 2  # the most a bot may download from Telegram (Bot API getFile)
DOC_EXTENSIONS = {".pdf", ".epub", ".docx", ".txt"}
CONVERT_EXTENSIONS = {".doc", ".mobi", ".azw", ".azw3", ".rtf", ".odt", ".fb2", ".djvu"}
DOC_FORMATS = "PDF, EPUB, DOCX or TXT"
TOO_BIG = ("⚠️ This file is larger than 20 MB, the most Telegram lets bots download. Upload it to Google Drive "
           "or Dropbox, share it as \"Anyone with the link\" and send me the link (or send a smaller version, e.g. "
           "an EPUB).")
DOWNLOAD_FAILED = "⚠️ Couldn't download the file from Telegram. Please send it again."
BOOK_MODES = {"whole": ("book", "📖 Whole book"), "short": ("chapters-short", "All chapters, short"),
              "each": ("chapters", "All chapters, one per message"), "pick": ("chapter-list", "Pick a chapter")}
CHAPTERS_PER_PAGE = 8


def _fmt_size(n: int) -> str:
    """File size as "850 KB" or "2.3 MB"."""
    return f"{n / 1024 ** 2:.1f} MB" if n >= 1024 ** 2 else f"{max(1, round(n / 1024))} KB"


def _book_menu(upload_id: int, chapters: bool = False, back: bool = True) -> InlineKeyboardMarkup:
    """The choice buttons under an upload: whole / by chapter, or the three chapter options (with ◀ Back to
    the first choice unless `back` is False, as under a whole-book summary)."""
    if not chapters:
        return InlineKeyboardMarkup([[InlineKeyboardButton("📖 Whole book", callback_data=f"book:{upload_id}:whole"),
                                      InlineKeyboardButton("📑 By chapter", callback_data=f"book:{upload_id}:chapters")]])
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("All chapters, short", callback_data=f"book:{upload_id}:short")],
        [InlineKeyboardButton("All chapters, one per message", callback_data=f"book:{upload_id}:each")],
        [InlineKeyboardButton("Pick a chapter", callback_data=f"book:{upload_id}:pick")],
    ] + ([[InlineKeyboardButton("◀ Back", callback_data=f"book:{upload_id}:back")]] if back else []))


def _chapter_list(upload_id: int, name: str, chapters: list[dict], page: int) -> tuple[str, InlineKeyboardMarkup]:
    """One page of the chapter list: the text and the buttons (chapters plus ◀ ▶)."""
    pages = max(1, -(-len(chapters) // CHAPTERS_PER_PAGE))
    page = min(max(page, 0), pages - 1)
    first = page * CHAPTERS_PER_PAGE
    rows = [[InlineKeyboardButton(f"{i + 1}. {re.sub(r'\s+', ' ', ch['title'])}"[:60],  # titles come from the file
                                  callback_data=f"book:{upload_id}:ch:{i}")]
            for i, ch in enumerate(chapters[first:first + CHAPTERS_PER_PAGE], first)]
    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton("◀", callback_data=f"book:{upload_id}:pg:{page - 1}"))
    if page < pages - 1:
        nav.append(InlineKeyboardButton("▶", callback_data=f"book:{upload_id}:pg:{page + 1}"))
    if nav:
        rows.append(nav)
    text = f"📑 {name[:80]}: pick a chapter" + (f" (page {page + 1} of {pages})" if pages > 1 else "")
    return text, InlineKeyboardMarkup(rows)


async def on_document(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles an uploaded file: checks its type and size, then asks how to summarize it."""
    if not await guard(update, ctx):
        return
    d = update.message.document
    name = d.file_name or "document"
    ext = Path(name).suffix.lower()
    if ext in CONVERT_EXTENSIONS:
        await update.message.reply_text(f"⚠️ I can't read {ext} files. Convert it to PDF or EPUB and send it again.")
        return
    if ext not in DOC_EXTENSIONS:
        await update.message.reply_text(f"⚠️ I can summarize {DOC_FORMATS} files, and YouTube or TikTok links.")
        return
    if not d.file_size or d.file_size > TG_DOWNLOAD_LIMIT:
        await update.message.reply_text(TOO_BIG)
        return
    upload_id = db.add_upload(update.effective_user.id, d.file_id, d.file_unique_id, name, d.file_size)
    await update.message.reply_text(f"📄 {name} ({_fmt_size(d.file_size)})\nHow should I summarize it?",
                                    reply_markup=_book_menu(upload_id))


async def _offer_link(update: Update, url: str, link: links.FileLink) -> None:
    """A Google Drive / Dropbox link: records it like an upload and asks how to summarize it.

    Nothing is downloaded yet: that happens in the job, once the user picks (and the limits allow it).
    """
    name = link.name or f"{link.service} file"
    ext = Path(link.name).suffix.lower()
    if ext in CONVERT_EXTENSIONS:
        await update.message.reply_text(f"⚠️ I can't read {ext} files. Convert it to PDF or EPUB and send it again.")
        return
    if link.name and ext not in DOC_EXTENSIONS:
        await update.message.reply_text(f"⚠️ I can summarize {DOC_FORMATS} files, and YouTube or TikTok links.")
        return
    upload_id = db.add_link_upload(update.effective_user.id, url, name)
    await update.message.reply_text(f"📄 {name} ({link.service})\nHow should I summarize it?",
                                    reply_markup=_book_menu(upload_id))


async def on_book_button(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the buttons under an upload and in its chapter list (`book:<upload id>:<action>[:<n>]`).

    Actions: whole / short / each / pick start a job; chapters / back switch the menu; pg:<n> pages the
    chapter list; ch:<n> summarizes one chapter. Callback data can be forged, so everything is re-checked:
    private chat, access, and that the upload is the user's own (or the user is an admin).
    """
    q = update.callback_query
    parts = (q.data or "").split(":")
    uid = q.from_user.id
    upload = db.get_upload(int(parts[1])) if len(parts) >= 3 and parts[1].isdigit() else None
    if (not _private(update) or access.state(uid) not in ("admin", "allowed") or upload is None
            or (upload["user_id"] != uid and not access.is_admin(uid))):
        await q.answer("This isn't available.")
        return
    action, arg = parts[2], (int(parts[3]) if len(parts) == 4 and parts[3].isdigit() else None)
    if action in ("chapters", "back"):
        await q.answer()
        await q.edit_message_reply_markup(_book_menu(upload["id"], chapters=action == "chapters"))
        return
    if action == "pg" and arg is not None:
        doc = db.get_document(upload["sha256"]) if upload["sha256"] else None
        if not doc or not doc["chapters"]:
            await q.answer("Please pick \"By chapter\" again.")
            return
        text, markup = _chapter_list(upload["id"], upload["name"], doc["chapters"], arg)
        await q.answer()
        await q.edit_message_text(text, reply_markup=markup)
        return
    if action not in (*BOOK_MODES, "ch") or (action == "ch" and arg is None):
        await q.answer("This isn't available.")
        return
    if action == "pick":
        # This upload was read already: its chapter list is stored, so show it now instead of queueing a
        # job (which would wait behind whatever runs, e.g. this book's other summaries). Nothing is
        # summarized, so nothing counts; the chapter tapped next is the request. Only for this very upload:
        # another user's upload of the same file reads it first, which keeps that a fresh-looking run.
        doc = db.get_document(upload["sha256"]) if upload["sha256"] else None
        if doc and doc["status"] == "done" and doc["chapters"]:
            text, markup = _chapter_list(upload["id"], upload["name"], doc["chapters"], 0)
            await q.answer()
            await ctx.bot.send_message(q.message.chat.id, text, reply_markup=markup)
            return
    request_id = None
    if action == "ch":  # the first chapter picked from a list continues the list's request
        request_id = db.waiting_request(uid, upload["sha256"]) if upload["sha256"] else None
    if refusal := _refusal(uid, new_request=request_id is None):
        await q.answer(refusal[:200], show_alert=True)
        return
    await q.answer()
    status = await ctx.bot.send_message(q.message.chat.id, _queued_message())
    mode, kind = ("pick", "chapter") if action == "ch" else (action, BOOK_MODES[action][0])
    if request_id:
        db.update_request(request_id, status="queued", kind=kind)
    await _start_job(uid, q.message.chat.id, status.message_id, f"📄 {upload['name']}", kind,
                     request_id=request_id, upload_id=upload["id"], book_mode=mode, chapter=arg)


async def on_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles plain messages: summarizes the first link in the text or media caption."""
    if not await guard(update, ctx):
        return
    text = update.message.text or update.message.caption or ""
    url = find_url(text)
    try:
        link = links.parse(url) if url else None
    except links.LinkError as e:
        await update.message.reply_text(str(e))
        return
    if link:
        await _offer_link(update, url, link)
        return
    await enqueue(update, url)


def command(**opts):
    """Makes a handler for a `/command <url>` that queues the link with the given job options.

    Args:
        **opts: Job options, e.g. `use_cache=False` (/again) or `transcript_only=True` (/transcript).

    Returns:
        An async command handler.
    """
    async def handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Queues the URL given as the command's argument, for allowed users."""
        if await guard(update, ctx):
            await enqueue(update, find_url(" ".join(ctx.args)), **opts)
    return handler


async def on_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles /start: the help text for allowed users, an access request for strangers.

    /start is the only way a stranger can file an access request (see `guard`).
    """
    user = update.effective_user
    db.touch_user(user.id, user.full_name, user.username)  # known users: refresh name (no-op otherwise)
    if await guard(update, ctx, request=True):
        await update.message.reply_text(HELP + (ADMIN_HELP if access.is_admin(update.effective_user.id) else ""))


async def on_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles /help: the help text, plus the admin commands for admins."""
    if await guard(update, ctx):
        await update.message.reply_text(HELP + (ADMIN_HELP if access.is_admin(update.effective_user.id) else ""))


# ---------- output ----------

def _secs(sec: float) -> str:
    """Formats a duration for the footer: "42 s" under a minute, "m:ss" from there on.

    Args:
        sec: Duration in seconds.

    Returns:
        The formatted duration.
    """
    sec = round(sec)
    return f"{sec} s" if sec < 60 else f"{sec // 60}:{sec % 60:02d}"


def details(r: pipeline.Result, waited: float = 0, reveal_cache: bool = True) -> str:
    """Builds the footer: how long each step took and what was used (transcript, frames, LLM).

    Args:
        r: The pipeline result.
        waited: Seconds the job waited in the queue; shown when it's 5 s or more.
        reveal_cache: Whether this user may learn the result came from the cache. A result with replay
            steps (see pipeline.plan_replay) shows them, like a fresh run's timings, so the footer matches how
            long the user actually waited.

    Returns:
        One or two lines of plain text (the caller HTML-escapes it).
    """
    stats = (r.summary or {}).get("_stats") or {}
    if r.replay_steps:
        steps = " · ".join(f"{name} {_secs(sec)}" for name, sec in r.replay_steps)
        timing = f"⏱ {_secs(r.replay_total)} total: {steps}"
        if waited >= 5:
            timing += f" (+ {_secs(waited)} waiting in queue)"
    elif r.cached and not reveal_cache:
        timing = ""  # never "from cache" (pipeline sets replay steps for these users; this is a backstop)
    elif r.cached:
        timing = "⚡ from cache" + (f" (first run took {_secs(stats['total'])})" if stats else "")
    elif stats:
        steps = " · ".join(f"{name} {_secs(sec)}" for name, sec in stats["steps"])
        timing = f"⏱ {_secs(stats['total'])} total: {steps}"
        if waited >= 5:
            timing += f" (+ {_secs(waited)} waiting in queue)"
    else:
        timing = ""
    source = {"captions": "YouTube captions", "tiktok-webvtt": "TikTok captions", "none": "none"}.get(
        r.transcript_source, r.transcript_source.replace("whisper-", "Whisper "))
    if r.transcript_source == "none":
        used = "📝 no speech found" if not r.meta.get("is_carousel") else "📝 photo post, no transcript"
    else:
        used = f"📝 transcript: {source}" + (f" ({r.language})" if r.language else "")
    if r.frames_used:
        used += " · 🖼 slides" if r.meta.get("is_carousel") else " · 🎞 video frames"
    # The model saved with the summary, not the current default: a cached summary may be from another model.
    used += f" · 🧠 {stats.get('llm') or summarize.llm_label()}"
    return "\n".join(filter(None, [timing, used]))


# Longest a model-written field may be. The prompt asks for much less; these only stop a broken or
# prompt-injected answer from turning into a flood of messages.
FIELD_LIMITS = {"title": 300, "clickbait_answer": 1000, "summary": 4000}


def _cap(text: str, limit: int) -> str:
    """Shortens text to at most `limit` characters, marking the cut with "…"."""
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def _escaped_pieces(raw_line: str) -> list[str]:
    """HTML-escapes one line of text, cut into pieces that each fit a message.

    The cut is made on the raw text, so it can never land inside an entity like `&amp;` (which Telegram
    rejects); pieces are sized by their escaped length, which can be up to 6x the raw length.
    """
    pieces, current, size = [], [], 0
    for ch in raw_line:
        esc = html.escape(ch)
        if size + len(esc) > TG_LIMIT:
            pieces.append("".join(current))
            current, size = [], 0
        current.append(esc)
        size += len(esc)
    pieces.append("".join(current))
    return pieces


def _pack(pieces: list[str]) -> list[str]:
    """Joins message pieces with newlines into as few messages as possible, each at most TG_LIMIT long.

    Every piece is complete HTML on its own (escaped text or a whole tag pair), so splitting between pieces
    keeps each message valid.
    """
    chunks, current = [], ""
    for piece in pieces:
        candidate = f"{current}\n{piece}" if current else piece
        if len(candidate) > TG_LIMIT and current:
            chunks.append(current)
            candidate = piece
        current = candidate
    if current.strip():
        chunks.append(current)
    return chunks


def render(r: pipeline.Result, waited: float = 0, reveal_cache: bool = True) -> list[str]:
    """Builds the reply in the Title / Clickbait answer / Summary layout, split to fit Telegram.

    All model and video text is HTML-escaped: messages are sent with parse_mode=HTML, and titles or
    summaries containing `<` or `&` would otherwise break parsing (or inject markup). Model fields are capped
    (FIELD_LIMITS) and messages are packed by their escaped length, so no message exceeds Telegram's limit.

    Args:
        r: The pipeline result (must have a summary).
        waited: Seconds the job waited in the queue (for the footer).
        reveal_cache: Whether this user may learn the result came from the cache (see `details`).

    Returns:
        One or more HTML messages, each at most TG_LIMIT characters.
    """
    s = r.summary

    def text(raw: str) -> list[str]:
        """Escaped pieces for a block of text, one or more per line."""
        return [piece for line in raw.split("\n") for piece in _escaped_pieces(line)]

    title = _cap(s.get("title") or r.meta.get("title", ""), FIELD_LIMITS["title"])
    if s.get("is_clickbait") and s.get("clickbait_answer"):
        answer = _cap(s["clickbait_answer"], FIELD_LIMITS["clickbait_answer"])
    else:
        answer = "✅ Not clickbait - the title matches the content."
    footer = _cap(details(r, waited, reveal_cache), 1000) + "\n" + r.url
    pieces = ["<b>Title:</b>", *text(title), "", "<b>Clickbait answer:</b>", *text(answer), "",
              "<b>Summary:</b>", *text(_cap(s.get("summary", ""), FIELD_LIMITS["summary"])), ""]
    # The footer is one italic piece: an <i> split across two messages would break both.
    pieces.append(f"<i>{html.escape(_cap(footer, 1500))}</i>")
    return _pack(pieces)


def _text(raw: str) -> list[str]:
    """Escaped message pieces for a block of text, one or more per line."""
    return [piece for line in raw.split("\n") for piece in _escaped_pieces(line)]


def _doc_details(r: documents.DocResult, waited: float, reveal_cache: bool) -> str:
    """The footer of a document summary: timings (like a fresh run for first-time requesters), the source."""
    if r.replay_steps:
        timing = f"⏱ {_secs(r.replay_total)} total: " + " · ".join(f"{n} {_secs(s)}" for n, s in r.replay_steps)
    elif r.cached:
        timing = "⚡ from cache" if reveal_cache else ""
    elif r.steps:
        timing = f"⏱ {_secs(r.total)} total: " + " · ".join(f"{n} {_secs(s)}" for n, s in r.steps)
    else:
        timing = ""
    if timing and waited >= 5:
        timing += f" (+ {_secs(waited)} waiting in queue)"
    fmt = (r.doc.get("format") or "").upper()
    pages = f"{r.doc.get('pages')} pages" if fmt == "PDF" else f"~{r.doc.get('pages')} pages"
    source = r.doc.get("text_source") or ""
    if source.startswith("ocr-"):
        engine_name = {"tesseract": "Tesseract", "rapidocr": "RapidOCR"}.get(source[4:], source[4:])
        fmt += f" · 🔍 OCR ({engine_name}, {ocr_names(r.doc.get('language'))})"
    used = f"📄 {r.name[:80]} · {fmt} · {pages} · 🧠 {r.llm}"
    return "\n".join(filter(None, [timing, used]))


def render_document(r: documents.DocResult, waited: float = 0, reveal_cache: bool = True) -> list[str]:
    """Builds the messages for a document summary, each at most TG_LIMIT characters.

    Whole book: Title / Author / Summary. All chapters short: one block per chapter, packed into as few
    messages as fit. One per message: a message per chapter. One chapter: its title and summary. The footer
    goes on the last message. Everything from the file or the model is escaped and capped.
    """
    footer = f"<i>{html.escape(_cap(_doc_details(r, waited, reveal_cache), 1500))}</i>"
    if r.kind == "book":
        b = r.book or {}
        pieces = ["<b>Title:</b>", *_text(_cap(b.get("title") or r.name, FIELD_LIMITS["title"]))]
        if b.get("author"):
            pieces += ["", "<b>Author:</b>", *_text(_cap(b["author"], FIELD_LIMITS["title"]))]
        pieces += ["", "<b>Summary:</b>", *_text(_cap(b.get("summary", ""), FIELD_LIMITS["summary"])), "", footer]
        return _pack(pieces)
    blocks = [[f"<b>{html.escape(_cap(title, 200))}</b>", *_text(_cap(summary, FIELD_LIMITS["summary"]))]
              for _, title, summary in r.chapters]
    if r.kind == "short":
        pieces = [f"<b>📑 {html.escape(r.name[:80])}</b>", ""]
        for block in blocks:
            pieces += [*block, ""]
        return _pack(pieces + [footer])
    if r.kind == "each" and len(blocks) > 1:
        n = len(blocks)
        out = []
        for k, block in enumerate(blocks, 1):
            head = [f"<b>{k}/{n}</b> " + block[0], *block[1:]]
            out += _pack(head + (["", footer] if k == n else []))
        return out
    return _pack([piece for block in blocks for piece in block] + ["", footer])


def _fmt_eta(sec: float) -> str:
    """Formats the remaining-time estimate, deliberately coarse so it reads as an estimate.

    Under a minute it's rounded *up* to 5 s (an optimistic ETA that keeps running out is worse than a
    slightly pessimistic one); under 10 min to half minutes; beyond that to whole minutes.

    Args:
        sec: Estimated seconds left.

    Returns:
        E.g. "15 s", "2.5 min", "12 min".
    """
    sec = max(int(sec), 0)
    if sec < 60:
        return f"{max(-(-sec // 5) * 5, 5)} s"  # round up to 5 s
    return f"{round(sec / 30) / 2:g} min" if sec < 600 else f"{round(sec / 60)} min"


class Progress:
    """Shows pipeline stages in the status message, with an ETA countdown refreshed every 15 s.

    Called from the worker thread. Edits run in call order (asyncio.Lock is FIFO), so a later
    stage is never overwritten by an earlier one, and no stage is dropped.
    """

    # Seconds between countdown refreshes. Edits count against Telegram's per-chat rate limit
    # (about one message per second), so the countdown stays well below it.
    TICK = 15

    def __init__(self, app: Application, loop: asyncio.AbstractEventLoop, job: Job):
        """Starts the countdown ticker for one job's status message.

        Args:
            app: The running application (for the bot).
            loop: The bot's event loop; edits are scheduled onto it from the worker thread.
            job: The job whose status message to edit.
        """
        self.app, self.loop, self.job = app, loop, job
        self.lock = asyncio.Lock()
        self.text, self.eta, self.eta_at, self.shown, self.last_edit = "", None, 0.0, "", 0.0
        self.started, self.closed = time.monotonic(), False
        self.ticker = asyncio.run_coroutine_threadsafe(self._tick(), loop)

    def __call__(self, text: str, eta: float | None = None) -> None:
        """Shows a new stage. Called by the pipeline from the worker thread.

        Args:
            text: The full status text (all lines).
            eta: Estimated seconds until the summary is ready, or None if unknown.
        """
        self.text, self.eta, self.eta_at = text, eta, time.monotonic()
        # The pipeline runs in a worker thread; Telegram calls must run on the bot's event loop. The stage is
        # passed along: rendering the *current* text instead would skip a stage that's replaced by the next
        # one before its edit runs.
        asyncio.run_coroutine_threadsafe(self._edit((text, eta, self.eta_at)), self.loop)

    async def close(self) -> None:
        """Stop updating; waits for an in-flight edit so it can't overwrite the final message."""
        self.ticker.cancel()
        async with self.lock:
            self.closed = True

    def _render(self, stage: tuple | None = None) -> str:
        """Builds the status text: a stage plus elapsed time and the ETA countdown.

        Args:
            stage: (text, eta, eta_at) of a specific stage; None = the current one (countdown refresh).

        Returns:
            The text for the status message.
        """
        text, eta, eta_at = stage or (self.text, self.eta, self.eta_at)
        elapsed = time.monotonic() - self.started
        line = f"⏱ {int(elapsed) // 60}:{int(elapsed) % 60:02d} elapsed"
        if eta is not None:
            # Count down from when the ETA was given, not from job start: each stage brings a new ETA.
            left = eta - (time.monotonic() - eta_at)
            line += f" · ~{_fmt_eta(left)} left" if left > 0 else " · taking longer than estimated…"
        return f"{text}\n\n{line}"

    async def _tick(self) -> None:
        """Refreshes the countdown every TICK seconds until cancelled."""
        while True:
            await asyncio.sleep(self.TICK)
            # Skip the refresh if a stage change just edited the message (TICK - 1 tolerates timer jitter):
            # that avoids a second, redundant edit right after it.
            if self.text and time.monotonic() - self.last_edit >= self.TICK - 1:
                await self._edit()

    async def _edit(self, stage: tuple | None = None) -> None:
        """Edits the status message, unless closed or unchanged.

        Args:
            stage: The stage to show (see _render); None = the current one.
        """
        # The lock serializes edits in call order, so a stage edit can't overtake a later one.
        async with self.lock:
            if self.closed:
                return
            text = self._render(stage)
            if text == self.shown:
                return
            try:
                await self.app.bot.edit_message_text(text, self.job.chat_id, self.job.status_id)
                self.shown, self.last_edit = text, time.monotonic()
            except RetryAfter as e:  # flood control: skip this edit, the next tick catches up
                log.warning("progress edit rate-limited for %ss", _retry_seconds(e))
            except TelegramError as e:
                # "message is not modified", the user blocked the bot, a network blip: a missed status edit
                # is harmless, and an exception here would kill the ticker task.
                log.debug("progress edit skipped: %s", e)


def _retry_seconds(e: RetryAfter) -> float:
    """Seconds Telegram asks us to wait. PTB returns an int today and a timedelta in a future version."""
    wait = e.retry_after
    return wait.total_seconds() if isinstance(wait, dt.timedelta) else float(wait)


class UserBlockedBot(Exception):
    """Telegram refuses to deliver to the user (they blocked the bot or deleted the chat)."""


def _check_not_cancelled(job: "Job | None") -> None:
    """Raises ProcCancelled if the job was cancelled: nothing more may be sent for it."""
    if job is not None and job.cancel_reason:
        raise proc.ProcCancelled("cancelled")


async def _send_with_retry(make_call, job: "Job | None" = None):
    """Runs one Telegram send, waiting out flood control once.

    Args:
        make_call: Zero-argument function returning the API coroutine (a coroutine can only be awaited once,
            so a retry needs a fresh one).
        job: The job the send belongs to. Checked before each attempt: a user removed while we waited out
            flood control must not get the result anyway.

    Raises:
        UserBlockedBot: Telegram says the user can't be reached.
        TelegramError: Any other Telegram failure, including a second flood-control refusal.
        proc.ProcCancelled: The job was cancelled before the send.
    """
    _check_not_cancelled(job)
    try:
        return await make_call()
    except RetryAfter as e:
        await asyncio.sleep(_retry_seconds(e))
        _check_not_cancelled(job)
        return await make_call()
    except Forbidden as e:
        raise UserBlockedBot(str(e))


CANCELLED = "✖️ Cancelled. You can send the link again any time."
MEMORY_RECHECK = 15  # seconds between memory re-checks for jobs set aside
# Jobs set aside because their transcription doesn't fit in memory right now; retried between other jobs.
_waiting_for_memory: list["Job"] = []


def _end_job(job: "Job") -> None:
    """Forgets a job that's completely done (finished, failed, cancelled or expired): frees its user's slot.

    Idempotent: several paths can end the same job (e.g. a cancel while the worker is reporting it), and only
    the first one may free the slot, or the user's counter would go wrong.
    """
    if _jobs.pop(job.request_id, None) is None:
        return
    _user_jobs[job.user_id] -= 1
    if _user_jobs[job.user_id] <= 0:
        del _user_jobs[job.user_id]


def _unpark(job: "Job") -> bool:
    """Takes a job off the memory-wait list; False if something else (a cancel) already did."""
    if job in _waiting_for_memory:
        _waiting_for_memory.remove(job)
        return True
    return False


async def _set_aside(app: Application, job: "Job", needed: int, duration: float = 0) -> None:
    """Parks a job that needs more memory than is free, and shows why, with a button to stop waiting."""
    if job.waiting_since is None:
        job.waiting_since = time.monotonic()
    job.memory_needed, job.audio_seconds = needed, duration
    _waiting_for_memory.append(job)
    db.update_request(job.request_id, status="queued")
    text = (f"🧠 Not enough free memory to transcribe this video right now (needs about "
            f"{needed / memory.GB:.1f} GB). Waiting up to {config.WHISPER_RAM_WAIT_MIN} min; other videos go "
            "first meanwhile.")
    button = InlineKeyboardMarkup([[InlineKeyboardButton("✖️ Don't wait, cancel",
                                                         callback_data=f"cancel:{job.request_id}")]])
    try:
        await app.bot.edit_message_text(text, job.chat_id, job.status_id, reply_markup=button)
    except TelegramError as e:
        log.debug("couldn't show the memory wait: %s", e)


def _memory_needed_now(job: "Job") -> int:
    """A waiting job's memory need, re-estimated: the model may have been loaded by another job since (it's
    then part of the bot's memory already and mustn't be counted again)."""
    if not job.audio_seconds:
        return job.memory_needed
    return memory.whisper_needs(job.audio_seconds, transcribe.is_loaded())


async def _next_job(app: Application) -> tuple["Job | None", bool]:
    """Picks the next job: a parked one whose memory now fits, else the next one from the queue.

    Parked jobs past their wait limit are failed here. While jobs are parked, waiting on the queue times out
    every MEMORY_RECHECK seconds so they're re-checked even when nothing new arrives.

    Returns:
        (job or None if nothing to do yet, whether it came from the queue).
    """
    now = time.monotonic()
    # Iterate over a snapshot, and re-check membership: the awaits below let a cancel take jobs off the list.
    for job in list(_waiting_for_memory):
        if job not in _waiting_for_memory:
            continue
        if job.cancel_reason:  # cancelled while parked: already reported
            _unpark(job)
            _end_job(job)
        elif now - job.waiting_since > config.WHISPER_RAM_WAIT_MIN * 60:
            _unpark(job)
            _end_job(job)  # bookkeeping before the await, so a concurrent cancel finds nothing to undo
            db.update_request(job.request_id, status="failed", error="waited too long for memory")
            await _fail(app, job, f"🧠 Still not enough free memory after {config.WHISPER_RAM_WAIT_MIN} min. "
                                  "Please try again later.")
        elif memory.fits_now(_memory_needed_now(job)):
            _unpark(job)
            return job, False
    try:
        timeout = MEMORY_RECHECK if _waiting_for_memory else None
        return await asyncio.wait_for(queue.get(), timeout), True
    except TimeoutError:
        return None, False


async def worker(app: Application) -> None:
    """Runs queued jobs one at a time, forever.

    Nothing may end this loop: if it stopped, the bot would keep acknowledging links ("⏳ Got it") but never
    process them, and systemd wouldn't notice since the process is still alive. So each job runs inside a
    last-resort `except Exception` (not BaseException: shutdown cancels this task and that must still work).

    Args:
        app: The running application.
    """
    global _running
    loop = asyncio.get_running_loop()
    while True:
        job, from_queue, parked = None, False, False
        # Everything, job selection included, is inside the safety net: a database hiccup or an unexpected
        # state while picking a job must not end the loop. Only Exception: shutdown's CancelledError must pass.
        try:
            job, from_queue = await _next_job(app)
            if job is None:
                continue
            if access.state(job.user_id) not in ("admin", "allowed") and not job.cancel_reason:
                # Access removed while queued (e.g. a path that didn't go through cancel_user_jobs).
                job.cancel_reason = ACCESS_REMOVED
                db.update_request(job.request_id, status="cancelled", error="cancelled: access removed")
                await _fail(app, job, ACCESS_REMOVED)
            if not job.cancel_reason:  # cancelled while queued: already reported, just skip it
                parked = await _run_job(app, loop, job)
        except Exception:
            log.exception("worker: job %s failed", job.request_id if job else "(selecting the next job)")
            await asyncio.sleep(1)  # don't spin if the failure repeats (e.g. the database is locked)
        finally:
            _running = None
            if job is not None and not parked:
                _end_job(job)
            if from_queue:
                queue.task_done()


async def _run_job(app: Application, loop: asyncio.AbstractEventLoop, job: Job) -> bool:
    """Runs one job: pipeline, reply, request bookkeeping. Failures are reported to the user and recorded.

    Returns:
        True if the job was set aside to wait for memory (it isn't done yet), else False.

    Args:
        app: The running application.
        loop: The event loop (for thread-safe status edits from the pipeline thread).
        job: The job to run.
    """
    global _running
    waited = time.monotonic() - job.queued_at
    proc.current_job_cancel.clear()
    _running = job
    try:
        progress = Progress(app, loop, job)
        try:
            if job.upload_id:
                result = await _run_document(app, job, progress)
            else:
                # The pipeline blocks (downloads, Whisper, LLM subprocesses): run it off the event loop.
                result = await asyncio.to_thread(
                    pipeline.run, job.url, progress, use_cache=job.use_cache, request_id=job.request_id,
                    backend=job.backend, model=job.model, transcript_only=job.transcript_only,
                    again_limit_user=None if access.is_admin(job.user_id) else job.user_id,
                    hide_cache_from=None if access.is_admin(job.user_id) else job.user_id)
        finally:
            # Before replying or deleting the status message: a late status edit must not land
            # after the final result.
            await progress.close()
        if job.cancel_reason:  # cancelled while the pipeline ran but finished before noticing (e.g. cached)
            raise proc.ProcCancelled("cancelled")
        if isinstance(result, documents.DocResult) and result.kind == "pick":
            # The status message becomes the chapter list (kept, not deleted); the first chapter picked from
            # it continues this request.
            text, markup = _chapter_list(job.upload_id, result.name, result.doc["chapters"], 0)
            await app.bot.edit_message_text(text, chat_id=job.chat_id, message_id=job.status_id, reply_markup=markup)
            db.update_request(job.request_id, status="waiting")
            return False
        if result.cached and result.replay_steps:  # a first-time requester (see pipeline.plan_replay)
            # Replayed in a separate task so the queue keeps moving meanwhile; the task ends the job.
            _delayed[job.request_id] = asyncio.create_task(_deliver_later(app, job, result, waited))
            return True
        await _deliver(app, job, result, waited)
        db.update_request(job.request_id, status="done", cached=int(result.cached))
        if not result.cached and not job.transcript_only:
            # Only fresh summaries teach the wait estimate: cache hits (~1 s) and transcripts would drag
            # the average far below what a queued link really waits for.
            if total := (result.summary or {}).get("_stats", {}).get("total"):
                stats.record("job", total)
        try:
            await app.bot.delete_message(job.chat_id, job.status_id)
        except TelegramError:
            pass  # the user deleted it already, or can't be reached: the result is delivered either way
    except Exception as e:
        # A cancel wins over whatever error it caused: e.g. a download killed by the cancel (or by systemd
        # at shutdown) must be reported as the cancel, not as "couldn't load this video".
        if job.cancel_reason or isinstance(e, proc.ProcCancelled):
            await _report_cancel(app, job)
            return False
        if isinstance(e, documents.NeedsOcr):
            await _ask_ocr(app, job, e)
            return False
        if isinstance(e, memory.NeedsMemory):
            await _set_aside(app, job, e.needed, e.duration)
            return True
        if isinstance(e, UserBlockedBot):
            db.update_request(job.request_id, status="failed", error="user blocked the bot")
            log.info("job %s: user %s blocked the bot", job.request_id, job.user_id)
        elif isinstance(e, (pipeline.PipelineError, UnsupportedURL)):
            # Expected failures: the message is written for the user; the technical detail is for the admin.
            detail = getattr(e, "detail", None)
            db.update_request(job.request_id, status="failed", error=(detail or str(e))[:500])
            if detail:
                log.warning("job %s failed: %s (%s)", job.request_id, e, detail)
            await _fail(app, job, str(e), detail if access.is_admin(job.user_id) else None)
            if isinstance(e, pipeline.Blocked):
                await _notify_admins_of_block(app, e)
        else:  # report, don't crash the worker
            log.exception("job failed: %s", job.url)
            detail = f"{type(e).__name__}: {e}"
            db.update_request(job.request_id, status="failed", error=f"unexpected: {detail}"[:500])
            if access.is_admin(job.user_id):
                await _fail(app, job, INTERNAL_ERROR, detail)
            else:
                await _fail(app, job, INTERNAL_ERROR_NOTIFIED)
                await _notify_admins_of_error(app, job, e)
    return False


_delayed: dict[int, asyncio.Task] = {}  # request id -> replay task of a cached summary (see _deliver_later)


def _ocr_status(uid: int) -> tuple[int, int | None, bool, float | None]:
    """A user's OCR use against their OCR limit, like _daily_status."""
    used, oldest = db.ocr_usage(uid, DAY)
    limit, is_default = access.ocr_limit(uid)
    frees_in = (oldest + DAY - time.time()) if limit and used >= limit and oldest else None
    return used, limit, is_default, frees_in


def _ocr_refusal(uid: int) -> str | None:
    """The refusal when a non-admin has used up today's OCR, else None."""
    used, limit, _, frees_in = _ocr_status(uid)
    if limit and used >= limit:
        return (f"⏳ This is a scanned document and needs text recognition (OCR). You've used today's {limit} "
                f"OCR documents; you can send more in about {_fmt_until(frees_in or 0)}.")
    return None


def _ocr_buttons(rid: int, over_cap: bool) -> InlineKeyboardMarkup:
    """Start/Cancel under an OCR confirmation, or OK/Ask an admin when the scan is over the page cap."""
    if over_cap:
        return InlineKeyboardMarkup([[InlineKeyboardButton("OK", callback_data=f"ocr:{rid}:ok"),
                                      InlineKeyboardButton("🙋 Ask admin for approval", callback_data=f"ocr:{rid}:ask")]])
    return InlineKeyboardMarkup([[InlineKeyboardButton("▶️ Start OCR", callback_data=f"ocr:{rid}:go"),
                                  InlineKeyboardButton("✖️ Cancel", callback_data=f"ocr:{rid}:no")]])


async def _ask_ocr(app: Application, job: Job, e: documents.NeedsOcr) -> None:
    """A scanned document: asks the user to confirm OCR (with the time it takes), or explains why it can't run.

    The request waits ("waiting") with its job remembered in ocr_holds; the Start button continues it, so it
    counts once toward the daily limit. Over OCR_MAX_PAGES the user can only accept or ask the admins.
    """
    db.save_ocr_hold(job.request_id, job.upload_id, job.book_mode, job.chapter, e.pages, e.seconds, e.language)
    hold = db.get_ocr_hold(job.request_id)
    if refusal := (None if access.is_admin(job.user_id) else _ocr_refusal(job.user_id)):
        db.update_request(job.request_id, status="failed", error="OCR limit reached")
        await _fail(app, job, refusal)
        return
    over_cap = e.pages > config.OCR_MAX_PAGES and not hold["approved"] and not access.is_admin(job.user_id)
    if over_cap:
        text = (f"⚠️ This scan has {e.pages} pages that need text recognition (OCR); I read up to "
                f"{config.OCR_MAX_PAGES} pages. Reading all {e.pages} would take about {_fmt_eta(e.seconds)}.")
    else:
        text = (f"🔍 This is a scanned document: {e.pages} pages need text recognition (OCR), about "
                f"{_fmt_eta(e.seconds)}, plus the summary. Start?")
    db.update_request(job.request_id, status="waiting")
    await app.bot.edit_message_text(text, chat_id=job.chat_id, message_id=job.status_id,
                                    reply_markup=_ocr_buttons(job.request_id, over_cap))


def ocr_names(langs: str | None) -> str:
    """Names of a Tesseract language string, e.g. "slv+eng" -> "Slovenian, English"."""
    return ocr.names((langs or "").split("+")) or "unknown"


async def on_ocr_button(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the user's OCR buttons (`ocr:<request id>:go|no|ok|ask`).

    go starts the OCR (continuing the request), no/ok drop it, ask sends the admins the details with
    Allow/Deny buttons (once per request). Everything is re-checked: chat, access, ownership, the request
    still waiting, the page cap, the OCR limit.
    """
    q = update.callback_query
    parts = (q.data or "").split(":")
    uid = q.from_user.id
    req = db.get_request(int(parts[1])) if len(parts) == 3 and parts[1].isdigit() else None
    hold = db.get_ocr_hold(req["id"]) if req else None
    if (not _private(update) or access.state(uid) not in ("admin", "allowed") or not hold
            or (req["user_id"] != uid and not access.is_admin(uid))):
        await q.answer("This isn't available.")
        return
    if req["status"] != "waiting":
        await q.answer("This request isn't waiting any more.")
        return
    action = parts[2]
    upload = db.get_upload(hold["upload_id"])
    if action in ("no", "ok"):
        db.update_request(req["id"], status="cancelled", error="OCR declined")
        await q.answer()
        await q.edit_message_text("✖️ Cancelled." if action == "no" else "OK, I won't read this scan.")
        return
    if action == "ask":
        if hold["asked"]:
            await q.answer("An admin has been asked already.")
            return
        db.update_ocr_hold(req["id"], asked=1)
        used, limit, _, _ = _ocr_status(req["user_id"])
        size = f" ({_fmt_size(upload['size'])})" if upload and upload["size"] else ""
        details = (f"🙋 {access.label(req['user_id'], db.get_user(req['user_id']))} asks to read a long scan:\n"
                   f"📄 {upload['name'][:100] if upload else '?'}{size}\n"
                   f"Pages needing OCR: {hold['pages']} (limit {config.OCR_MAX_PAGES})\n"
                   f"Language: {ocr_names(hold['language'])}\n"
                   f"Estimated OCR time: about {_fmt_eta(hold['seconds'])}, plus the summary\n"
                   f"Their OCR use today: {used}" + (f"/{limit}" if limit else ""))
        buttons = InlineKeyboardMarkup([[InlineKeyboardButton("✅ Allow", callback_data=f"ocradm:{req['id']}:yes"),
                                         InlineKeyboardButton("❌ Deny", callback_data=f"ocradm:{req['id']}:no")]])
        for admin in access.ADMINS:
            try:
                await ctx.bot.send_message(admin, details, reply_markup=buttons)
            except TelegramError as e:
                log.warning("couldn't ask admin %s: %s", admin, e)
        await q.answer()
        await q.edit_message_text("🙋 I asked an admin; I'll let you know their answer.")
        return
    if action != "go":
        await q.answer("This isn't available.")
        return
    if hold["pages"] > config.OCR_MAX_PAGES and not hold["approved"] and not access.is_admin(uid):
        await q.answer("This needs an admin's approval first.", show_alert=True)
        return
    refusal = None if access.is_admin(req["user_id"]) else _ocr_refusal(req["user_id"])
    if refusal := refusal or _refusal(req["user_id"], new_request=False):
        await q.answer(refusal[:200], show_alert=True)
        return
    await q.answer()
    await q.edit_message_text(_queued_message())
    db.update_request(req["id"], status="queued")
    await _start_job(req["user_id"], q.message.chat.id, q.message.message_id, req["url"], req["kind"],
                     request_id=req["id"], upload_id=hold["upload_id"], book_mode=hold["mode"],
                     chapter=hold["chapter"], ocr_ok=True)


async def on_ocr_admin_button(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles an admin's Allow/Deny on a long-scan request (`ocradm:<request id>:yes|no`)."""
    q = update.callback_query
    parts = (q.data or "").split(":")
    if not _private(update) or not access.is_admin(q.from_user.id):
        await q.answer("Only admins can do this.")
        return
    req = db.get_request(int(parts[1])) if len(parts) == 3 and parts[1].isdigit() else None
    hold = db.get_ocr_hold(req["id"]) if req else None
    if not hold or req["status"] != "waiting":
        await q.answer("Already handled.")
        await q.edit_message_reply_markup(None)
        return
    by = q.from_user.full_name
    if parts[2] == "yes":
        db.update_ocr_hold(req["id"], approved=1)
        await ctx.bot.send_message(
            req["user_id"], f"✅ An admin allowed reading all {hold['pages']} pages (about "
                            f"{_fmt_eta(hold['seconds'])}, plus the summary). Start?",
            reply_markup=_ocr_buttons(req["id"], False))
        await q.edit_message_text(f"{q.message.text}\n\n✅ Allowed by {by}.")
    else:
        db.update_request(req["id"], status="cancelled", error="long OCR denied")
        await ctx.bot.send_message(req["user_id"], "❌ An admin declined reading this long scan.")
        await q.edit_message_text(f"{q.message.text}\n\n❌ Denied by {by}.")
    await q.answer()


async def _download_link(upload: dict, path: Path, head: str, progress: "Progress") -> dict:
    """Downloads a Drive/Dropbox-linked document (in a thread: it can take minutes), showing the progress.

    Returns:
        The upload row, with the file's real name once the service revealed it.

    Raises:
        documents.DocumentError: Not shared publicly, too large, or the download failed.
    """
    link = links.parse(upload["source_url"])  # parsed again: only the bot-built URL is ever fetched

    def shown(done: int, total: int | None) -> None:
        """Download progress in the status message."""
        of = f" of {_fmt_size(total)}" if total else ""
        progress(f"{head}\n📥 Downloading the file… {_fmt_size(done)}{of}", None)

    try:
        name = await asyncio.to_thread(links.download, link, path, shown)
    except links.LinkError as e:
        raise documents.DocumentError(str(e), e.detail)
    if name and name != upload["name"]:
        db.set_upload_name(upload["id"], name)
        upload = db.get_upload(upload["id"])
    return upload


async def _run_document(app: Application, job: Job, progress: "Progress") -> documents.DocResult:
    """Runs a document job: downloads the file from Telegram unless its text is stored, then documents.run.

    Raises:
        documents.DocumentError: The file can't be downloaded, read or summarized.
    """
    upload = db.get_upload(job.upload_id)
    if upload is None:
        raise documents.DocumentError("⚠️ Please send the file again.", f"upload {job.upload_id} missing")
    workdir = config.DATA_DIR / "work" / f"doc_{job.request_id}"
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)
    try:
        path = None
        doc = db.get_document(upload["sha256"]) if upload["sha256"] else None
        if not doc or doc["status"] != "done":
            head = f"📄 {upload['name'][:80]}"
            progress(f"{head}\n📥 Downloading the file…", None)
            path = workdir / "upload"  # no extension: the format is told from the bytes
            if upload["source_url"]:
                upload = await _download_link(upload, path, head, progress)
            else:
                try:
                    tg_file = await app.bot.get_file(upload["file_id"])
                    await tg_file.download_to_drive(path)
                except BadRequest as e:
                    raise documents.DocumentError(TOO_BIG if "too big" in str(e).lower() else DOWNLOAD_FAILED, str(e))
                except TelegramError as e:
                    raise documents.DocumentError(DOWNLOAD_FAILED, str(e))
            if job.cancel_reason:
                raise proc.ProcCancelled("cancelled")
        return await asyncio.to_thread(
            documents.run, upload, job.book_mode, job.chapter, path, workdir, progress, backend=job.backend,
            model=job.model, request_id=job.request_id,
            hide_cache_from=None if access.is_admin(job.user_id) else job.user_id, ocr_confirmed=job.ocr_ok)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)  # only the extracted text is kept


async def _deliver_later(app: Application, job: Job, result: pipeline.Result, waited: float) -> None:
    """Replays a cached answer (summary or transcript) like a fresh run, for a user who mustn't learn it was
    cached.

    Shows the steps of result.replay_steps for their (already scaled) times, then delivers. Runs beside the
    worker (which doesn't wait for it), and stops if the job is cancelled meanwhile. Always ends the job.
    """
    try:
        stats = (result.summary or {}).get("_stats") or {}
        progress = Progress(app, asyncio.get_running_loop(), job)
        meta = getattr(result, "meta", None) or {}
        if isinstance(result, documents.DocResult):
            head = result.head
        elif meta.get("is_carousel"):
            head = f"🖼 {meta.get('title', '')[:80]} (photo post)"
        else:
            head = f"🎬 {meta.get('title', '')[:80]} ({pipeline._fmt_duration(meta.get('duration') or 0)})"
        llm = getattr(result, "llm", "") or stats.get("llm") or summarize.llm_label(job.backend, job.model or "")
        remaining = result.replay_total
        try:
            for name, sec in result.replay_steps:
                if job.cancel_reason:
                    return
                progress(f"{head}\n{pipeline.replay_stage(name, llm)}", remaining)
                await asyncio.sleep(sec)
                remaining -= sec
        finally:
            await progress.close()
        if job.cancel_reason:
            return
        await _deliver(app, job, result, waited)
        db.update_request(job.request_id, status="done", cached=1)
        try:
            await app.bot.delete_message(job.chat_id, job.status_id)
        except TelegramError:
            pass
    except proc.ProcCancelled:
        await _report_cancel(app, job)  # cancelled between parts of the summary
    except UserBlockedBot:
        db.update_request(job.request_id, status="failed", error="user blocked the bot")
    except Exception:
        log.exception("replaying cached result for job %s failed", job.request_id)
    finally:
        _delayed.pop(job.request_id, None)
        _end_job(job)


async def _deliver(app: Application, job: Job, result: pipeline.Result, waited: float) -> None:
    """Sends a finished job's result: the transcript file, or the summary message(s).

    Raises:
        UserBlockedBot: The user can't be reached.
        TelegramError: Telegram refused the message.
    """
    bot_ = app.bot
    if isinstance(result, documents.DocResult):
        reveal = access.is_admin(job.user_id) or db.user_saw_video(job.user_id, "document", result.video_id,
                                                                    job.request_id)
        chunks = render_document(result, waited, reveal)
        # Under a whole-book summary: the chapter options, for a reader who wants more detail.
        more = (_book_menu(job.upload_id, chapters=True, back=False)
                if result.kind == "book" and len(result.doc.get("chapters") or []) > 1 else None)
        for k, chunk in enumerate(chunks, 1):
            await _send_with_retry(lambda chunk=chunk, k=k: bot_.send_message(
                job.chat_id, chunk, parse_mode=ParseMode.HTML, disable_web_page_preview=True,
                reply_markup=more if k == len(chunks) else None), job)
        return
    if job.transcript_only:
        if not result.transcript:
            await _send_with_retry(lambda: bot_.send_message(job.chat_id, "No transcript available for this video."),
                                   job)
        else:
            data = result.transcript.encode()
            # A long transcript is a sizeable upload; PTB's default 5 s write timeout is too short for it.
            await _send_with_retry(lambda: bot_.send_document(
                job.chat_id, io.BytesIO(data), filename=f"{result.video_id}.txt",
                caption=f"Transcript ({result.transcript_source})", write_timeout=60), job)
        return
    # Only admins, or the user who asked for this video before, may learn it was cached: otherwise it
    # would reveal what other users submit.
    reveal = access.is_admin(job.user_id) or db.user_saw_video(
        job.user_id, result.platform, result.video_id, job.request_id)
    for chunk in render(result, waited, reveal_cache=reveal):
        await _send_with_retry(lambda chunk=chunk: bot_.send_message(
            job.chat_id, chunk, parse_mode=ParseMode.HTML, disable_web_page_preview=True), job)


INTERNAL_ERROR = "⚠️ Something went wrong on the bot's side."
INTERNAL_ERROR_NOTIFIED = INTERNAL_ERROR + " The admin has been notified."
ADMIN_ERROR_NOTICE_EVERY = 600  # seconds: one notice per error type, so a recurring bug can't flood the admins
_admin_error_noticed: dict[str, float] = {}  # error type -> when the admins were last told


async def _notify_admins_of_error(app: Application, job: Job, e: Exception) -> None:
    """Tells the admins about an unexpected error in another user's job (at most once per type per 10 min)."""
    kind = type(e).__name__
    now = time.monotonic()
    if now - _admin_error_noticed.get(kind, -ADMIN_ERROR_NOTICE_EVERY) < ADMIN_ERROR_NOTICE_EVERY:
        return
    _admin_error_noticed[kind] = now
    who = access.label(job.user_id, db.get_user(job.user_id))
    text = f"⚠️ Unexpected error in a job for {who}\nLink: {job.url}\nDetails: {kind}: {str(e)[:500]}"
    for admin in access.ADMINS:
        try:
            await app.bot.send_message(admin, text, disable_web_page_preview=True)
        except TelegramError as err:
            log.warning("couldn't notify admin %s: %s", admin, err)


BLOCK_NOTICE_EVERY = 6 * 3600  # seconds: a block lasts hours; one notice per platform in that time is enough
_block_noticed: dict[str, float] = {}  # platform -> when the admins were last told


async def _notify_admins_of_block(app: Application, e: "pipeline.Blocked") -> None:
    """Tells the admins that YouTube/TikTok is blocking the server (at most once per platform per 6 h).

    They may need to wait it out, update yt-dlp, or set up cookies/a proxy.
    """
    now = time.monotonic()
    if now - _block_noticed.get(e.platform, -BLOCK_NOTICE_EVERY) < BLOCK_NOTICE_EVERY:
        return
    _block_noticed[e.platform] = now
    text = (f"🚫 {e.platform} is blocking downloads from this server. Users are being told to try later.\n"
            f"Raw error: {(e.detail or '')[:500]}\nOptions: wait, update yt-dlp, or add cookies/a proxy.")
    for admin in access.ADMINS:
        try:
            await app.bot.send_message(admin, text, disable_web_page_preview=True)
        except TelegramError as err:
            log.warning("couldn't notify admin %s: %s", admin, err)


async def _fail(app: Application, job: Job, msg: str, detail: str | None = None) -> None:
    """Shows an error in place of the job's status message. Never raises.

    Falls back to a new message if the status message can't be edited (e.g. it was deleted). If the user
    can't be reached at all, the error is only logged: raising here would escape the job's error handling.

    Args:
        app: The running application.
        job: The failed job.
        msg: The message for the user (one of the expected texts, never a raw error).
        detail: Technical detail, only passed when the requester is an admin (they maintain the bot).
    """
    text = f"⚠️ {msg}" if msg[:1].isalnum() else msg  # messages with their own emoji keep it
    if detail:
        text += f"\n\nDetails: {detail[:800]}"
    try:
        try:
            await app.bot.edit_message_text(text, job.chat_id, job.status_id)
        except BadRequest:
            await app.bot.send_message(job.chat_id, text)
    except TelegramError as e:
        log.warning("couldn't report the failure of job %s: %s", job.request_id, e)


# The chat's "Menu" button and "/" autocomplete, per user: strangers only see /start, approved users the
# normal commands, admins also /users.
STRANGER_COMMANDS = [BotCommand("start", "Request access to this bot")]
# /start isn't listed for approved users: they don't need it, and its menu entry says "request access".
USER_COMMANDS = [
    BotCommand("help", "How to use the bot"),
    BotCommand("again", "Summarize again, ignoring the cache: /again <url>"),
    BotCommand("transcript", "Get the raw transcript as a file: /transcript <url>"),
    BotCommand("history", "Your recent requests"),
    BotCommand("models", "Show or choose the AI: Codex or Claude, then the model"),
    BotCommand("limit", "Your daily limit"),
]
# Admin-specific commands first; /history is re-described ("all users"), so the user version is dropped.
ADMIN_COMMANDS = [BotCommand("users", "Manage users: allow, remove, unblock"),
                  BotCommand("history", "Recent requests from all users"),
                  BotCommand("limit", "Daily limits: show or set"),
                  BotCommand("ocrlang", "Languages for scanned documents (OCR)")] + [
    c for c in USER_COMMANDS if c.command not in ("history", "limit")]


async def sync_commands(bot, uid: int) -> None:
    """Sets the command menu for one user's chat to match their access.

    Args:
        bot: The Telegram bot.
        uid: Telegram user id (equal to the private chat id).
    """
    st = access.state(uid)
    scope = BotCommandScopeChat(uid)
    try:
        if st == "admin":
            await bot.set_my_commands(ADMIN_COMMANDS, scope=scope)
        elif st == "allowed":
            await bot.set_my_commands(USER_COMMANDS, scope=scope)
        else:
            await bot.delete_my_commands(scope=scope)  # back to the default (stranger) menu
    except (BadRequest, Forbidden) as e:  # user hasn't opened a chat with the bot
        log.warning("couldn't set commands for %s: %s", uid, e)


async def post_init(app: Application) -> None:
    """Startup: registers the command menus and starts the worker and the update checker.

    Menus are re-synced for every admin and allowed user on each start, so the menus match the database
    and the code's command lists even after either changed while the bot was down.

    Args:
        app: The application being started.
    """
    if missing := ocr.missing_languages():
        log.warning("OCR (%s) has no model for installed language(s) %s: add them again with /ocrlang add",
                    ocr.engine(), ", ".join(missing))
    await app.bot.set_my_commands(STRANGER_COMMANDS, scope=BotCommandScopeDefault())
    for uid in [*access.ADMINS, *(u["id"] for u in access.all_users()["allowed"])]:
        await sync_commands(app.bot, uid)
    # Keep references in bot_data: post_stop cancels them, and asyncio only weakly references tasks.
    # Before the worker starts: nothing is running, so everything in the job folders is a leftover, and any
    # request still marked queued/processing was lost with the previous run's in-memory queue.
    await asyncio.to_thread(pipeline.cleanup_leftovers)
    if stale := db.fail_stale_requests():
        log.info("marked %d request(s) from before the restart as failed", stale)
    app.bot_data["worker"] = asyncio.create_task(worker(app))
    app.bot_data["update_checker"] = asyncio.create_task(update_checker(app))
    log.info("bot ready; admins: %s, allowed users: %d", sorted(access.ADMINS) or "NONE (setup mode)",
             len(access.all_users()["allowed"]))


UPDATE_CHECK_EVERY = 12 * 3600  # seconds; CLI releases come at most a few times a week


async def update_checker(app: Application) -> None:
    """Tells the admins when a newer Codex / Claude Code is out (once per new version).

    The last version announced per tool is remembered in stats.json, so a restart or the next check
    doesn't repeat the same notification.

    Args:
        app: The running application.
    """
    await asyncio.sleep(60)  # let startup finish
    while True:
        try:
            # Runs `--version` and `npm view` subprocesses: keep them off the event loop.
            for u in await asyncio.to_thread(updates.check):
                key = f"update-notified:{u['tool']}"
                if stats.recall(key) == u["latest"]:
                    continue
                text = (f"⬆️ {html.escape(u['tool'])} update available: {u['installed']} → {u['latest']}\n"
                        f"New models may need it. On the bot's machine run:\n"
                        f"<pre>{html.escape(u['command'])}</pre>")  # tap to copy, and no @openai link
                for admin in access.ADMINS:
                    try:
                        await app.bot.send_message(admin, text, parse_mode=ParseMode.HTML)
                    except (BadRequest, Forbidden) as e:
                        log.warning("couldn't notify admin %s: %s", admin, e)
                stats.remember(key, u["latest"])
                log.info("notified admins: %s %s -> %s", u["tool"], u["installed"], u["latest"])
        except Exception:
            # Network or npm hiccups must not kill the loop; try again next round.
            log.exception("update check failed")
        await asyncio.sleep(UPDATE_CHECK_EVERY)


STOPPED = "⏹ The bot was stopped before your summary was ready. Please send the link again later."
SHUTDOWN_WAIT = 30  # seconds to let the running job stop and report (systemd's TimeoutStopSec is 60)


async def post_stop(app: Application) -> None:
    """Shutdown: tells everyone still waiting that the bot stopped, stops their work, then the tasks.

    PTB calls this after polling stopped but before the bot's connection closes, so messages can still be
    sent. Every unfinished job is cancelled with STOPPED (queued, waiting for memory or replaying ones are
    reported at once; the running one has its programs killed and the worker reports it). The running job
    gets SHUTDOWN_WAIT seconds to do so (an API call or a model load can't be interrupted sooner); if it
    doesn't make it, it's reported from here. Only then are the worker and update checker cancelled (without
    that, the worker blocked in `queue.get()` was destroyed while pending: "Event loop is closed" errors).

    Args:
        app: The application being stopped.
    """
    running = _running
    for job in list(_jobs.values()):
        await _cancel_job(app, job, STOPPED)
    deadline = time.monotonic() + SHUTDOWN_WAIT
    while running is not None and _running is running and time.monotonic() < deadline:
        await asyncio.sleep(0.2)
    if running is not None and running.request_id in _jobs:  # the worker didn't get to report it in time
        _end_job(running)
        await _report_cancel(app, running)
    for name in ("worker", "update_checker"):
        if task := app.bot_data.get(name):
            task.cancel()


async def on_error(update: object, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Logs exceptions raised by handlers (python-telegram-bot's error-handler hook).

    Args:
        update: The update being handled, if any.
        ctx: Handler context; `ctx.error` is the exception.
    """
    log.error("update handling failed", exc_info=ctx.error)


def add_handlers(app: Application) -> None:
    """Registers the bot's update handlers and error handler on an application.

    Separate from main() so tests can build an application with exactly the bot's handlers.
    """
    app.add_error_handler(on_error)
    # New messages only: an edited message would re-run a command (and edited ones carry no
    # update.message, which the handlers use to reply).
    # Private chats only (see _private).
    new = filters.UpdateType.MESSAGE & filters.ChatType.PRIVATE
    app.add_handler(ChatMemberHandler(on_my_chat_member, ChatMemberHandler.MY_CHAT_MEMBER))
    for name, handler in [("start", on_start), ("help", on_help), ("again", command(use_cache=False)),
                          ("transcript", command(transcript_only=True)), ("users", on_users),
                          ("history", on_history), ("models", on_models), ("limit", on_limit),
                          ("ocrlang", on_ocrlang)]:
        app.add_handler(CommandHandler(name, handler, filters=new))
    # Order matters: the first matching handler wins, so the picker's `llm:` buttons must be registered
    # before the catch-all admin-button handler.
    app.add_handler(CallbackQueryHandler(on_llm_button, pattern=r"^llm:"))
    app.add_handler(CallbackQueryHandler(on_cancel_button, pattern=r"^cancel:"))
    app.add_handler(CallbackQueryHandler(on_book_button, pattern=r"^book:"))
    app.add_handler(CallbackQueryHandler(on_ocr_button, pattern=r"^ocr:"))
    app.add_handler(CallbackQueryHandler(on_ocr_admin_button, pattern=r"^ocradm:"))
    app.add_handler(CallbackQueryHandler(on_button, pattern=r"^(allow|block|remove):"))
    # Before on_message: a file sent with a caption is a document, not a message with a link.
    app.add_handler(MessageHandler(new & filters.Document.ALL, on_document))
    app.add_handler(MessageHandler(new & (filters.TEXT | filters.CAPTION) & ~filters.COMMAND
                                   & ~filters.Document.ALL, on_message))


def main() -> None:
    """Builds the application, registers the handlers and runs long polling until stopped.

    Raises:
        SystemExit: If TELEGRAM_BOT_TOKEN isn't set.
    """
    if not config.TELEGRAM_BOT_TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set (.env)")
    config.secure_files()
    app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).post_init(post_init).post_stop(post_stop).build()
    add_handlers(app)
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
