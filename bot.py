"""Telegram bot: send a YouTube/TikTok link, get back title, clickbait answer and summary."""
import asyncio
import html
import io
import logging
import time
from dataclasses import dataclass

from telegram import (BotCommand, BotCommandScopeChat, BotCommandScopeDefault, InlineKeyboardButton,
                      InlineKeyboardMarkup, Update, User)
from telegram.constants import ParseMode
from telegram.error import BadRequest, Forbidden, RetryAfter
from telegram.ext import (Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler,
                          filters)

import access
from summarizer import config, pipeline
from summarizer.urls import UnsupportedURL, find_url

logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("bot")

TG_LIMIT = 4096
HELP = (
    "Send me a YouTube or TikTok link and I'll reply with the title, an answer to any clickbait, "
    "and a summary.\n\n"
    "/frames <url> - also look at video frames (on-screen tables, text)\n"
    "/noframes <url> - transcript only, skip frames\n"
    "/again <url> - ignore the cache and summarize again\n"
    "/transcript <url> - send the raw transcript as a file"
)
ADMIN_HELP = "\n\nAdmin:\n/users - list users; allow, remove, or unblock them"


@dataclass
class Job:
    url: str
    chat_id: int
    status_id: int
    force_frames: bool | None = None
    use_cache: bool = True
    transcript_only: bool = False


queue: asyncio.Queue[Job] = asyncio.Queue()


def _info(user: User) -> dict:
    return {"name": user.full_name, "username": user.username}


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
    info = access.set_state(user.id, "pending", _info(user))
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
        rows = [[InlineKeyboardButton(f"{text}: {info.get('name') or uid}", callback_data=f"{act}:{uid}")
                 for text, act in actions[st]] for uid, info in users[st].items()]
        lines = "\n".join(f"• {access.label(uid, info)}" for uid, info in users[st].items())
        await update.message.reply_text(f"{titles[st]}\n{lines}", reply_markup=InlineKeyboardMarkup(rows))


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
    await queue.put(Job(url, update.effective_chat.id, status.message_id, **opts))


async def on_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not await guard(update, ctx):
        return
    text = update.message.text or update.message.caption or ""
    words = text.lower().split()
    force = True if "frames" in words else False if "noframes" in words else None
    await enqueue(update, find_url(text), force_frames=force)


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

def render(r: pipeline.Result) -> list[str]:
    """Build the reply in the Title / Clickbait answer / Summary layout, split to fit Telegram."""
    s = r.summary
    e = html.escape
    head = f"<b>Title:</b>\n{e(s['title'] or r.meta['title'])}\n\n<b>Clickbait answer:</b>\n"
    head += e(s["clickbait_answer"]) if s["is_clickbait"] and s["clickbait_answer"] else "✅ Not clickbait - the title matches the content."
    head += "\n\n<b>Summary:</b>\n"
    source = r.transcript_source + (f", {r.language}" if r.language else "")
    foot = f"\n\n<i>{e(r.url)} · transcript: {e(source)}" + (" · frames used" if r.frames_used else "") \
           + (" · cached" if r.cached else "") + "</i>"

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
        try:
            progress = Progress(app, loop, job)
            try:
                result = await asyncio.to_thread(
                    pipeline.run, job.url, progress, force_frames=job.force_frames,
                    use_cache=job.use_cache, transcript_only=job.transcript_only)
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
                for chunk in render(result):
                    await app.bot.send_message(job.chat_id, chunk, parse_mode=ParseMode.HTML,
                                               disable_web_page_preview=True)
            await app.bot.delete_message(job.chat_id, job.status_id)
        except (pipeline.PipelineError, UnsupportedURL) as e:
            await _fail(app, job, str(e))
        except Exception as e:  # report, don't crash the worker
            log.exception("job failed: %s", job.url)
            await _fail(app, job, f"Unexpected error: {e}")
        finally:
            queue.task_done()


async def _fail(app: Application, job: Job, msg: str) -> None:
    try:
        await app.bot.edit_message_text(f"⚠️ {msg}", job.chat_id, job.status_id)
    except BadRequest:
        await app.bot.send_message(job.chat_id, f"⚠️ {msg}")


COMMANDS = [
    BotCommand("help", "How to use the bot"),
    BotCommand("frames", "Summarize and look at video frames: /frames <url>"),
    BotCommand("noframes", "Summarize from the transcript only: /noframes <url>"),
    BotCommand("again", "Summarize again, ignoring the cache: /again <url>"),
    BotCommand("transcript", "Get the raw transcript as a file: /transcript <url>"),
    BotCommand("start", "Start / request access"),
]
ADMIN_COMMANDS = [BotCommand("users", "Manage users: allow, remove, unblock")]


async def post_init(app: Application) -> None:
    # The chat's "Menu" button and "/" autocomplete. Admins get /users on top.
    await app.bot.set_my_commands(COMMANDS, scope=BotCommandScopeDefault())
    for admin in access.ADMINS:
        try:
            await app.bot.set_my_commands(ADMIN_COMMANDS + COMMANDS, scope=BotCommandScopeChat(admin))
        except BadRequest as e:  # admin hasn't opened a chat with the bot yet
            log.warning("couldn't set admin commands for %s: %s", admin, e)
    app.bot_data["worker"] = asyncio.create_task(worker(app))
    log.info("bot ready; admins: %s, allowed users: %d", sorted(access.ADMINS) or "NONE (setup mode)",
             len(access.all_users()["allowed"]))


async def post_stop(app: Application) -> None:
    if task := app.bot_data.get("worker"):
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
    app.add_handler(CommandHandler("frames", command(force_frames=True)))
    app.add_handler(CommandHandler("noframes", command(force_frames=False)))
    app.add_handler(CommandHandler("again", command(use_cache=False)))
    app.add_handler(CommandHandler("transcript", command(transcript_only=True)))
    app.add_handler(CommandHandler("users", on_users))
    app.add_handler(CallbackQueryHandler(on_button))
    app.add_handler(MessageHandler((filters.TEXT | filters.CAPTION) & ~filters.COMMAND, on_message))
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
