"""Telegram bot: send a YouTube/TikTok link, get back title, clickbait answer and summary."""
import asyncio
import collections
import datetime as dt
import html
import io
import logging
import os
import time
from dataclasses import dataclass

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
from summarizer import config, db, pipeline, proc, stats, summarize, updates
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
    "/models - show or choose the AI (Codex or Claude) and model"
)
ADMIN_HELP = ("\n\nAdmin:\n/users - list users; allow, remove, or unblock them\n"
              "/history - recent requests from all users (who sent what, cache hits)")


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


# A single queue drained by a single worker: jobs run one at a time, so Whisper (CPU-heavy) never runs
# twice in parallel, and two users' LLM conversations can't interleave.
queue: asyncio.Queue[Job] = asyncio.Queue()


ACCESS_REMOVED = "⛔ Your access to this bot was removed, so this request was cancelled."

# Every job not yet finished, by request id, so a user's jobs can be found and cancelled. The worker runs one
# job at a time; _running is that one (its programs are stopped through proc.current_job_cancel).
_jobs: dict[int, "Job"] = {}
_running: "Job | None" = None


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
        job.cancel_reason = reason
        if job is _running:
            proc.current_job_cancel.set()  # the worker reports it when the pipeline stops
        else:
            db.update_request(job.request_id, status="cancelled", error="cancelled: " + reason[:200])
            await _fail(app, job, reason)
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

    admins = "\n".join(f"• {access.label(u['id'], u)}: {llm(u)}" for u in users["admin"])
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
    titles = {"allowed": "✅ Allowed", "pending": "⏳ Pending", "blocked": "⛔ Blocked"}
    for st in access.STATES:
        if not users[st]:
            continue
        rows = [[InlineKeyboardButton(f"{text}: {u['name'] or u['id']}", callback_data=f"{act}:{u['id']}")
                 for text, act in actions[st]] for u in users[st]]
        lines = "\n".join(f"• {access.label(u['id'], u)}" + (f": {llm(u)}" if st == "allowed" else "")
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
    icons = {"done": "✅", "failed": "⚠️", "queued": "⏳", "processing": "⏳", "cancelled": "✖️"}
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
        await update.message.reply_text("Send me a YouTube or TikTok link.")
        return
    try:
        check_url(url)  # no network: a bad link is refused before it gets a queue slot or a request row
    except UnsupportedURL as e:
        await update.message.reply_text(f"⚠️ {e}")
        return
    uid = update.effective_user.id
    if not access.is_admin(uid):
        if _user_jobs[uid] >= config.MAX_QUEUED_PER_USER:
            await update.message.reply_text(
                f"⏳ You already have {_user_jobs[uid]} videos in the queue. Send this one again when one of "
                "them is done.")
            return
        if queue.qsize() >= config.MAX_QUEUE:
            await update.message.reply_text(
                f"⏳ The bot is busy right now ({queue.qsize()} videos queued). Please try again in a few minutes.")
            return
    ahead = queue.qsize()
    status = await update.message.reply_text(
        "⏳ Got it" + (f", queued (position {ahead + 1})" if ahead else ", working…"))
    kind = "transcript" if opts.get("transcript_only") else "again" if opts.get("use_cache") is False else "summary"
    req = db.add_request(uid, url, kind)  # logged the moment the link arrives
    b, m, is_default = _current_llm(uid)
    # Users on the defaults pass None, so their jobs follow the default (and its later changes)
    # instead of pinning whatever the default resolves to right now.
    b, m = (None, None) if is_default else (b, m)
    _user_jobs[uid] += 1
    job = Job(url, update.effective_chat.id, status.message_id, queued_at=time.monotonic(),
              user_id=uid, request_id=req, backend=b, model=m, **opts)
    _jobs[req] = job
    await queue.put(job)


async def on_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles plain messages: summarizes the first link in the text or media caption."""
    if not await guard(update, ctx):
        return
    text = update.message.text or update.message.caption or ""
    await enqueue(update, find_url(text))


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
        reveal_cache: Whether this user may learn the result came from the cache. When False, a cached
            result shows no timing line at all: the original run's timings would show it was cached,
            revealing that someone else submitted the video.

    Returns:
        One or two lines of plain text (the caller HTML-escapes it).
    """
    stats = (r.summary or {}).get("_stats") or {}
    if r.cached and not reveal_cache:
        timing = ""
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


def render(r: pipeline.Result, waited: float = 0, reveal_cache: bool = True) -> list[str]:
    """Builds the reply in the Title / Clickbait answer / Summary layout, split to fit Telegram.

    All model and video text is HTML-escaped: messages are sent with parse_mode=HTML, and titles or
    summaries containing `<` or `&` would otherwise break parsing (or inject markup).

    Args:
        r: The pipeline result (must have a summary).
        waited: Seconds the job waited in the queue (for the footer).
        reveal_cache: Whether this user may learn the result came from the cache (see `details`).

    Returns:
        One or more HTML messages, each at most TG_LIMIT characters.
    """
    s = r.summary
    e = html.escape
    head = f"<b>Title:</b>\n{e(s['title'] or r.meta['title'])}\n\n<b>Clickbait answer:</b>\n"
    head += e(s["clickbait_answer"]) if s["is_clickbait"] and s["clickbait_answer"] else "✅ Not clickbait - the title matches the content."
    head += "\n\n<b>Summary:</b>\n"
    foot = "\n\n<i>" + e(details(r, waited, reveal_cache)) + f"\n{e(r.url)}</i>"

    body = e(s["summary"])
    if len(head) + len(body) + len(foot) <= TG_LIMIT:
        return [head + body + foot]
    # Too long for one message: split the summary on line boundaries.
    chunks, cur = [], head
    for line in body.split("\n"):
        # A single line over the limit is cut hard; the 100-character margin leaves room for the
        # newline and the chunk's other content.
        while len(line) > TG_LIMIT - 100:  # pathological single line
            chunks.append(cur)
            cur, line = line[:TG_LIMIT - 100], line[TG_LIMIT - 100:]
        if len(cur) + len(line) + 1 > TG_LIMIT:
            chunks.append(cur)
            cur = ""
        cur += line + "\n"
    if len(cur) + len(foot) > TG_LIMIT:
        chunks.append(cur)
        cur = ""
    chunks.append(cur.rstrip("\n") + foot)
    return [c for c in chunks if c.strip()]


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


async def _send_with_retry(make_call):
    """Runs one Telegram send, waiting out flood control once.

    Args:
        make_call: Zero-argument function returning the API coroutine (a coroutine can only be awaited once,
            so a retry needs a fresh one).

    Raises:
        UserBlockedBot: Telegram says the user can't be reached.
        TelegramError: Any other Telegram failure, including a second flood-control refusal.
    """
    try:
        return await make_call()
    except RetryAfter as e:
        await asyncio.sleep(_retry_seconds(e))
        return await make_call()
    except Forbidden as e:
        raise UserBlockedBot(str(e))


async def worker(app: Application) -> None:
    """Runs queued jobs one at a time, forever.

    Nothing may end this loop: if it stopped, the bot would keep acknowledging links ("⏳ Got it") but never
    process them, and systemd wouldn't notice since the process is still alive. So each job runs inside a
    last-resort `except Exception` (not BaseException: shutdown cancels this task and that must still work).

    Args:
        app: The running application.
    """
    loop = asyncio.get_running_loop()
    while True:
        job = await queue.get()
        try:
            if access.state(job.user_id) not in ("admin", "allowed") and not job.cancel_reason:
                # Access removed while queued (e.g. a path that didn't go through cancel_user_jobs).
                job.cancel_reason = ACCESS_REMOVED
                db.update_request(job.request_id, status="cancelled", error="cancelled: access removed")
                await _fail(app, job, ACCESS_REMOVED)
            if not job.cancel_reason:  # cancelled while queued: already reported, just skip it
                await _run_job(app, loop, job)
        except Exception:
            log.exception("worker: job %s failed while reporting its result", job.request_id)
        finally:
            global _running
            _running = None
            _jobs.pop(job.request_id, None)
            _user_jobs[job.user_id] -= 1
            if _user_jobs[job.user_id] <= 0:
                del _user_jobs[job.user_id]
            queue.task_done()


async def _run_job(app: Application, loop: asyncio.AbstractEventLoop, job: Job) -> None:
    """Runs one job: pipeline, reply, request bookkeeping. Failures are reported to the user and recorded.

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
            # The pipeline blocks (downloads, Whisper, LLM subprocesses): run it off the event loop.
            result = await asyncio.to_thread(
                pipeline.run, job.url, progress, use_cache=job.use_cache, request_id=job.request_id,
                backend=job.backend, model=job.model, transcript_only=job.transcript_only,
                again_limit_user=None if access.is_admin(job.user_id) else job.user_id)
        finally:
            # Before replying or deleting the status message: a late status edit must not land
            # after the final result.
            await progress.close()
        await _deliver(app, job, result, waited)
        db.update_request(job.request_id, status="done", cached=int(result.cached))
        try:
            await app.bot.delete_message(job.chat_id, job.status_id)
        except TelegramError:
            pass  # the user deleted it already, or can't be reached: the result is delivered either way
    except proc.ProcCancelled:
        reason = job.cancel_reason or "Cancelled."
        db.update_request(job.request_id, status="cancelled", error="cancelled: " + reason[:200])
        await _fail(app, job, reason)
    except UserBlockedBot:
        db.update_request(job.request_id, status="failed", error="user blocked the bot")
        log.info("job %s: user %s blocked the bot", job.request_id, job.user_id)
    except (pipeline.PipelineError, UnsupportedURL) as e:
        # Expected failures: the message is written for the user; the technical detail is for the admin.
        detail = getattr(e, "detail", None)
        db.update_request(job.request_id, status="failed", error=(detail or str(e))[:500])
        if detail:
            log.warning("job %s failed: %s (%s)", job.request_id, e, detail)
        await _fail(app, job, str(e), detail if access.is_admin(job.user_id) else None)
        if isinstance(e, pipeline.Blocked):
            await _notify_admins_of_block(app, e)
    except Exception as e:  # report, don't crash the worker
        if job.cancel_reason:  # e.g. an API call that ended oddly because of the cancel: report the cancel
            db.update_request(job.request_id, status="cancelled", error="cancelled: " + job.cancel_reason[:200])
            await _fail(app, job, job.cancel_reason)
            return
        log.exception("job failed: %s", job.url)
        detail = f"{type(e).__name__}: {e}"
        db.update_request(job.request_id, status="failed", error=f"unexpected: {detail}"[:500])
        if access.is_admin(job.user_id):
            await _fail(app, job, INTERNAL_ERROR, detail)
        else:
            await _fail(app, job, INTERNAL_ERROR_NOTIFIED)
            await _notify_admins_of_error(app, job, e)


async def _deliver(app: Application, job: Job, result: pipeline.Result, waited: float) -> None:
    """Sends a finished job's result: the transcript file, or the summary message(s).

    Raises:
        UserBlockedBot: The user can't be reached.
        TelegramError: Telegram refused the message.
    """
    bot_ = app.bot
    if job.transcript_only:
        if not result.transcript:
            await _send_with_retry(lambda: bot_.send_message(job.chat_id, "No transcript available for this video."))
        else:
            data = result.transcript.encode()
            # A long transcript is a sizeable upload; PTB's default 5 s write timeout is too short for it.
            await _send_with_retry(lambda: bot_.send_document(
                job.chat_id, io.BytesIO(data), filename=f"{result.video_id}.txt",
                caption=f"Transcript ({result.transcript_source})", write_timeout=60))
        return
    # Only admins, or the user who asked for this video before, may learn it was cached: otherwise it
    # would reveal what other users submit.
    reveal = access.is_admin(job.user_id) or db.user_saw_video(
        job.user_id, result.platform, result.video_id, job.request_id)
    for chunk in render(result, waited, reveal_cache=reveal):
        await _send_with_retry(lambda chunk=chunk: bot_.send_message(
            job.chat_id, chunk, parse_mode=ParseMode.HTML, disable_web_page_preview=True))


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
    text = msg if msg.startswith(("⚠️", "⏳", "🔴", "🔒", "🔞", "🌍", "🚫", "⛔")) else f"⚠️ {msg}"
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
]
# Admin-specific commands first; /history is re-described ("all users"), so the user version is dropped.
ADMIN_COMMANDS = [BotCommand("users", "Manage users: allow, remove, unblock"),
                  BotCommand("history", "Recent requests from all users")] + [
    c for c in USER_COMMANDS if c.command != "history"]


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
    await app.bot.set_my_commands(STRANGER_COMMANDS, scope=BotCommandScopeDefault())
    for uid in [*access.ADMINS, *(u["id"] for u in access.all_users()["allowed"])]:
        await sync_commands(app.bot, uid)
    # Keep references in bot_data: post_stop cancels them, and asyncio only weakly references tasks.
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


async def post_stop(app: Application) -> None:
    """Shutdown: cancels the background tasks so they don't outlive the event loop.

    Without this, the worker blocked in `queue.get()` was destroyed while pending, which logged
    "Event loop is closed" errors on every stop.

    Args:
        app: The application being stopped.
    """
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
                          ("history", on_history), ("models", on_models)]:
        app.add_handler(CommandHandler(name, handler, filters=new))
    # Order matters: the first matching handler wins, so the picker's `llm:` buttons must be registered
    # before the catch-all admin-button handler.
    app.add_handler(CallbackQueryHandler(on_llm_button, pattern=r"^llm:"))
    app.add_handler(CallbackQueryHandler(on_button))
    app.add_handler(MessageHandler(new & (filters.TEXT | filters.CAPTION) & ~filters.COMMAND, on_message))


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
