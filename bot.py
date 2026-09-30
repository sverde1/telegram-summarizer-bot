"""Telegram bot: send a YouTube/TikTok link, get back title, clickbait answer and summary."""
import asyncio
import html
import io
import logging
import time
from dataclasses import dataclass

from telegram import (BotCommand, BotCommandScopeChat, BotCommandScopeDefault, InlineKeyboardButton,
                      InlineKeyboardMarkup, Update)
from telegram.constants import ParseMode
from telegram.error import BadRequest, Forbidden, RetryAfter
from telegram.ext import (Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler,
                          filters)

import access
from summarizer import config, db, pipeline, stats, summarize, updates
from summarizer.urls import UnsupportedURL, find_url

logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("bot")

TG_LIMIT = 4096
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


queue: asyncio.Queue[Job] = asyncio.Queue()


async def guard(update: Update, ctx: ContextTypes.DEFAULT_TYPE, request: bool = False) -> bool:
    """True if the user may use the bot. With request=True (/start), an unknown user's access
    request is sent to the admins; otherwise they're only told how to ask."""
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
    if st == "blocked":
        log.info("ignored blocked user %s", user.id)
        return False
    if st == "pending":
        if msg:
            await msg.reply_text("⏳ Your access request is waiting for the admin's approval.")
        return False
    if not request:
        if msg:
            await msg.reply_text("🔒 This is a private bot. Send /start to request access.")
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
    if not access.is_admin(update.effective_user.id):
        return
    users = access.all_users()
    if not any(users.values()):
        await update.message.reply_text("No users besides you yet. When someone messages the bot, "
                                        "you'll get an access request here.")
        return
    actions = {"allowed": [("🗑 Remove", "remove")], "pending": [("✅ Allow", "allow"), ("❌ Deny", "block")],
               "blocked": [("↩️ Unblock", "remove")]}
    titles = {"allowed": "✅ Allowed", "pending": "⏳ Pending", "blocked": "⛔ Blocked"}
    for st in access.STATES:
        if not users[st]:
            continue
        rows = [[InlineKeyboardButton(f"{text}: {u['name'] or u['id']}", callback_data=f"{act}:{u['id']}")
                 for text, act in actions[st]] for u in users[st]]
        lines = "\n".join(f"• {access.label(u['id'], u)}" for u in users[st])
        await update.message.reply_text(f"{titles[st]}\n{lines}", reply_markup=InlineKeyboardMarkup(rows))


def _current_llm(uid: int) -> tuple[str, str, bool]:
    """(backend, model, is_default) this user's summaries use."""
    backend, model = db.get_user_llm(uid)
    if backend not in summarize.available_backends():
        backend, model = None, None  # their choice was uninstalled: fall back
    b = backend or config.LLM_BACKEND
    return b, model or summarize.default_model(b), not backend and not model


def _llm_home(uid: int) -> tuple[str, InlineKeyboardMarkup]:
    """Step 1 of /models: pick a provider."""
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


def _llm_models(uid: int, backend: str) -> tuple[str, InlineKeyboardMarkup]:
    """Step 2 of /models: pick a model of one provider."""
    b, m, _ = _current_llm(uid)
    try:
        models = summarize.list_models(backend)
    except summarize.SummaryError as e:
        return f"⚠️ {e}", InlineKeyboardMarkup([[InlineKeyboardButton("⬅️ Back", callback_data="llm:home")]])
    default = summarize.default_model(backend)
    lines = [f"🧠 {summarize.BACKEND_NAMES[backend]} models:", ""]
    rows = []
    for model in models:
        mark = "✓ " if (backend, model["id"]) == (b, m) else ""
        tag = " (default)" if model["id"] == default else ""
        lines.append(f"{mark}{model['id']}{tag}" + (f": {model['description']}" if model["description"] else ""))
        rows.append([InlineKeyboardButton(f"{mark}{model['name'] or model['id']}{tag}",
                                          callback_data=f"llm:m:{backend}:{model['id']}"[:64])])
    rows.append([InlineKeyboardButton("⬅️ Back", callback_data="llm:home")])
    return "\n".join(lines), InlineKeyboardMarkup(rows)


def _set_llm(uid: int, backend: str | None, model: str | None) -> str:
    """Validate and store a user's choice; returns the confirmation text."""
    if backend is None:
        db.set_user_llm(uid, None, None)
        b = config.LLM_BACKEND
        return f"✅ Back to the default: {summarize.BACKEND_NAMES[b]} · {summarize.default_model(b)}"
    if backend not in summarize.available_backends():
        return f"⚠️ {backend} isn't available on this bot."
    try:
        ids = [m["id"] for m in summarize.list_models(backend)]
    except summarize.SummaryError as e:
        return f"⚠️ {e}"
    if model not in ids:
        return f"Unknown {summarize.BACKEND_NAMES[backend]} model “{model}”. Available: {', '.join(ids)}"
    db.set_user_llm(uid, backend, model)
    return (f"✅ Your summaries now use {summarize.BACKEND_NAMES[backend]} · {model}. "
            "Each video gets a summary written by this model (kept separately from other models' summaries).")


async def on_models(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if await guard(update, ctx):
        text, buttons = _llm_home(update.effective_user.id)
        await update.message.reply_text(text, reply_markup=buttons)


async def on_llm_button(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    q = update.callback_query
    uid = q.from_user.id
    if access.state(uid) not in ("admin", "allowed"):
        await q.answer()
        return
    parts = (q.data or "").split(":", 3)  # llm:home | llm:default | llm:b:<backend> | llm:m:<backend>:<model>
    if parts[1] == "b":
        text, buttons = _llm_models(uid, parts[2])
        await q.answer()
    elif parts[1] in ("m", "default"):
        reply = _set_llm(uid, parts[2], parts[3]) if parts[1] == "m" else _set_llm(uid, None, None)
        await q.answer(reply[:200])
        text, buttons = _llm_models(uid, parts[2]) if parts[1] == "m" else _llm_home(uid)
    else:
        text, buttons = _llm_home(uid)
        await q.answer()
    try:
        await q.edit_message_text(text, reply_markup=buttons)
    except BadRequest:  # unchanged
        pass


async def on_history(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Recent requests: admins see everyone's (with who sent them), users only their own."""
    if not await guard(update, ctx):
        return
    uid = update.effective_user.id
    admin = access.is_admin(uid)
    rows = db.recent_requests(None if admin else uid, limit=20)
    if not rows:
        await update.message.reply_text("No requests yet.")
        return
    icons = {"done": "✅", "failed": "⚠️", "queued": "⏳", "processing": "⏳"}
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
    q = update.callback_query
    if not access.is_admin(q.from_user.id):
        await q.answer("Admins only.")
        return
    action, _, uid_s = (q.data or "").partition(":")
    if action not in ("allow", "block", "remove") or not uid_s.isdigit():
        await q.answer()
        return
    uid = int(uid_s)
    if access.is_admin(uid):
        await q.answer("That's an admin (configured in .env).")
        return
    new = {"allow": "allowed", "block": "blocked", "remove": None}[action]
    info = access.set_state(uid, new) or {}
    who = access.label(uid, info)
    done = {"allow": f"✅ Allowed {who}", "block": f"⛔ Denied and blocked {who}",
            "remove": f"🗑 Removed {who}"}[action]
    log.info("admin %s: %s", q.from_user.id, done)
    await q.answer(done[:200])
    await q.edit_message_text(done)
    await sync_commands(ctx.bot, uid)
    if action == "allow":
        try:
            await ctx.bot.send_message(uid, "✅ You now have access. Send me a YouTube or TikTok link.\n\n" + HELP)
        except (BadRequest, Forbidden) as e:
            log.warning("couldn't notify user %s: %s", uid, e)


async def enqueue(update: Update, url: str | None, **opts) -> None:
    if not url:
        await update.message.reply_text("Send me a YouTube or TikTok link.")
        return
    ahead = queue.qsize()
    status = await update.message.reply_text(
        "⏳ Got it" + (f", queued (position {ahead + 1})" if ahead else ", working…"))
    kind = "transcript" if opts.get("transcript_only") else "again" if opts.get("use_cache") is False else "summary"
    uid = update.effective_user.id
    req = db.add_request(uid, url, kind)  # logged the moment the link arrives
    b, m, is_default = _current_llm(uid)
    b, m = (None, None) if is_default else (b, m)
    await queue.put(Job(url, update.effective_chat.id, status.message_id, queued_at=time.monotonic(),
                        user_id=uid, request_id=req, backend=b, model=m, **opts))


async def on_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not await guard(update, ctx):
        return
    text = update.message.text or update.message.caption or ""
    await enqueue(update, find_url(text))


def command(**opts):
    async def handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if await guard(update, ctx):
            await enqueue(update, find_url(" ".join(ctx.args)), **opts)
    return handler


async def on_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if await guard(update, ctx, request=True):
        await update.message.reply_text(HELP + (ADMIN_HELP if access.is_admin(update.effective_user.id) else ""))


async def on_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if await guard(update, ctx):
        await update.message.reply_text(HELP + (ADMIN_HELP if access.is_admin(update.effective_user.id) else ""))


# ---------- output ----------

def _secs(sec: float) -> str:
    sec = round(sec)
    return f"{sec} s" if sec < 60 else f"{sec // 60}:{sec % 60:02d}"


def details(r: pipeline.Result, waited: float = 0, reveal_cache: bool = True) -> str:
    """Footer: how long each step took and what was used."""
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
    used += f" · 🧠 {stats.get('llm') or summarize.llm_label()}"
    return "\n".join(filter(None, [timing, used]))


def render(r: pipeline.Result, waited: float = 0, reveal_cache: bool = True) -> list[str]:
    """Build the reply in the Title / Clickbait answer / Summary layout, split to fit Telegram."""
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
    sec = max(int(sec), 0)
    if sec < 60:
        return f"{max(-(-sec // 5) * 5, 5)} s"  # round up to 5 s
    return f"{round(sec / 30) / 2:g} min" if sec < 600 else f"{round(sec / 60)} min"


class Progress:
    """Shows pipeline stages in the status message, with an ETA countdown refreshed every 15 s.

    Called from the worker thread. Edits run in call order (asyncio.Lock is FIFO), so a later
    stage is never overwritten by an earlier one, and no stage is dropped.
    """

    TICK = 15

    def __init__(self, app: Application, loop: asyncio.AbstractEventLoop, job: Job):
        self.app, self.loop, self.job = app, loop, job
        self.lock = asyncio.Lock()
        self.text, self.eta, self.eta_at, self.shown, self.last_edit = "", None, 0.0, "", 0.0
        self.started, self.closed = time.monotonic(), False
        self.ticker = asyncio.run_coroutine_threadsafe(self._tick(), loop)

    def __call__(self, text: str, eta: float | None = None) -> None:
        self.text, self.eta, self.eta_at = text, eta, time.monotonic()
        asyncio.run_coroutine_threadsafe(self._edit(), self.loop)

    async def close(self) -> None:
        """Stop updating; waits for an in-flight edit so it can't overwrite the final message."""
        self.ticker.cancel()
        async with self.lock:
            self.closed = True

    def _render(self) -> str:
        elapsed = time.monotonic() - self.started
        line = f"⏱ {int(elapsed) // 60}:{int(elapsed) % 60:02d} elapsed"
        if self.eta is not None:
            left = self.eta - (time.monotonic() - self.eta_at)
            line += f" · ~{_fmt_eta(left)} left" if left > 0 else " · taking longer than estimated…"
        return f"{self.text}\n\n{line}"

    async def _tick(self) -> None:
        while True:
            await asyncio.sleep(self.TICK)
            if self.text and time.monotonic() - self.last_edit >= self.TICK - 1:
                await self._edit()

    async def _edit(self) -> None:
        async with self.lock:
            if self.closed:
                return
            text = self._render()
            if text == self.shown:
                return
            try:
                await self.app.bot.edit_message_text(text, self.job.chat_id, self.job.status_id)
                self.shown, self.last_edit = text, time.monotonic()
            except RetryAfter as e:  # flood control: skip this edit, the next tick catches up
                log.warning("progress edit rate-limited for %ss", e.retry_after)
            except BadRequest as e:  # "message is not modified" etc.
                log.debug("progress edit skipped: %s", e)


async def worker(app: Application) -> None:
    loop = asyncio.get_running_loop()
    while True:
        job = await queue.get()
        waited = time.monotonic() - job.queued_at
        try:
            progress = Progress(app, loop, job)
            try:
                result = await asyncio.to_thread(
                    pipeline.run, job.url, progress, use_cache=job.use_cache, request_id=job.request_id,
                    backend=job.backend, model=job.model, transcript_only=job.transcript_only)
            finally:
                await progress.close()
            if job.transcript_only:
                if not result.transcript:
                    await app.bot.send_message(job.chat_id, "No transcript available for this video.")
                else:
                    doc = io.BytesIO(result.transcript.encode())
                    await app.bot.send_document(job.chat_id, doc, filename=f"{result.video_id}.txt",
                                                caption=f"Transcript ({result.transcript_source})")
            else:
                # Only admins, or the user who asked for this video before, may learn it was cached:
                # otherwise it would reveal what other users submit.
                reveal = access.is_admin(job.user_id) or db.user_saw_video(
                    job.user_id, result.platform, result.video_id, job.request_id)
                for chunk in render(result, waited, reveal_cache=reveal):
                    await app.bot.send_message(job.chat_id, chunk, parse_mode=ParseMode.HTML,
                                               disable_web_page_preview=True)
            db.update_request(job.request_id, status="done", cached=int(result.cached))
            await app.bot.delete_message(job.chat_id, job.status_id)
        except (pipeline.PipelineError, UnsupportedURL) as e:
            db.update_request(job.request_id, status="failed", error=str(e)[:500])
            await _fail(app, job, str(e))
        except Exception as e:  # report, don't crash the worker
            log.exception("job failed: %s", job.url)
            db.update_request(job.request_id, status="failed", error=f"unexpected: {e}"[:500])
            await _fail(app, job, f"Unexpected error: {e}")
        finally:
            queue.task_done()


async def _fail(app: Application, job: Job, msg: str) -> None:
    try:
        await app.bot.edit_message_text(f"⚠️ {msg}", job.chat_id, job.status_id)
    except BadRequest:
        await app.bot.send_message(job.chat_id, f"⚠️ {msg}")


# The chat's "Menu" button and "/" autocomplete, per user: strangers only see /start, approved users the
# normal commands, admins also /users.
STRANGER_COMMANDS = [BotCommand("start", "Request access to this bot")]
USER_COMMANDS = [
    BotCommand("help", "How to use the bot"),
    BotCommand("again", "Summarize again, ignoring the cache: /again <url>"),
    BotCommand("transcript", "Get the raw transcript as a file: /transcript <url>"),
    BotCommand("history", "Your recent requests"),
    BotCommand("models", "Show or choose the AI: Codex or Claude, then the model"),
]
ADMIN_COMMANDS = [BotCommand("users", "Manage users: allow, remove, unblock"),
                  BotCommand("history", "Recent requests from all users")] + [
    c for c in USER_COMMANDS if c.command != "history"]


async def sync_commands(bot, uid: int) -> None:
    """Set the command menu for one user's chat to match their access."""
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
    await app.bot.set_my_commands(STRANGER_COMMANDS, scope=BotCommandScopeDefault())
    for uid in [*access.ADMINS, *(u["id"] for u in access.all_users()["allowed"])]:
        await sync_commands(app.bot, uid)
    app.bot_data["worker"] = asyncio.create_task(worker(app))
    app.bot_data["update_checker"] = asyncio.create_task(update_checker(app))
    log.info("bot ready; admins: %s, allowed users: %d", sorted(access.ADMINS) or "NONE (setup mode)",
             len(access.all_users()["allowed"]))


UPDATE_CHECK_EVERY = 12 * 3600


async def update_checker(app: Application) -> None:
    """Tell the admins when a newer Codex / Claude Code is out (once per new version)."""
    await asyncio.sleep(60)  # let startup finish
    while True:
        try:
            for u in await asyncio.to_thread(updates.check):
                key = f"update-notified:{u['tool']}"
                if stats.recall(key) == u["latest"]:
                    continue
                text = (f"⬆️ {u['tool']} update available: {u['installed']} → {u['latest']}\n"
                        f"New models may need it. On the bot's machine run:\n{u['command']}")
                for admin in access.ADMINS:
                    try:
                        await app.bot.send_message(admin, text)
                    except (BadRequest, Forbidden) as e:
                        log.warning("couldn't notify admin %s: %s", admin, e)
                stats.remember(key, u["latest"])
                log.info("notified admins: %s %s -> %s", u["tool"], u["installed"], u["latest"])
        except Exception:
            log.exception("update check failed")
        await asyncio.sleep(UPDATE_CHECK_EVERY)


async def post_stop(app: Application) -> None:
    for name in ("worker", "update_checker"):
        if task := app.bot_data.get(name):
            task.cancel()


async def on_error(update: object, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    log.error("update handling failed", exc_info=ctx.error)


def main() -> None:
    if not config.TELEGRAM_BOT_TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set (.env)")
    app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).post_init(post_init).post_stop(post_stop).build()
    app.add_error_handler(on_error)
    app.add_handler(CommandHandler("start", on_start))
    app.add_handler(CommandHandler("help", on_help))
    app.add_handler(CommandHandler("again", command(use_cache=False)))
    app.add_handler(CommandHandler("transcript", command(transcript_only=True)))
    app.add_handler(CommandHandler("users", on_users))
    app.add_handler(CommandHandler("history", on_history))
    app.add_handler(CommandHandler("models", on_models))
    app.add_handler(CallbackQueryHandler(on_llm_button, pattern=r"^llm:"))
    app.add_handler(CallbackQueryHandler(on_button))
    app.add_handler(MessageHandler((filters.TEXT | filters.CAPTION) & ~filters.COMMAND, on_message))
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
