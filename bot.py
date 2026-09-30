"""Telegram bot: send a YouTube/TikTok link, get back title, clickbait answer and summary."""
import asyncio
import html
import io
import logging
import time
from dataclasses import dataclass

from telegram import Update
from telegram.constants import ParseMode
from telegram.error import BadRequest, RetryAfter
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

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


@dataclass
class Job:
    url: str
    chat_id: int
    status_id: int
    force_frames: bool | None = None
    use_cache: bool = True
    transcript_only: bool = False


queue: asyncio.Queue[Job] = asyncio.Queue()


def allowed(update: Update) -> bool:
    uid = update.effective_user.id if update.effective_user else None
    return uid in config.ALLOWED_USER_IDS


async def guard(update: Update) -> bool:
    if allowed(update):
        return True
    user = update.effective_user
    log.warning("rejected user id=%s username=%s", user and user.id, user and user.username)
    if not config.ALLOWED_USER_IDS and update.message:  # setup mode: tell the owner their id
        await update.message.reply_text(
            f"Setup: your Telegram user id is {user.id}. Add ALLOWED_USER_IDS={user.id} to .env "
            "and restart the bot.")
    return False


async def enqueue(update: Update, url: str | None, **opts) -> None:
    if not url:
        await update.message.reply_text("Send me a YouTube or TikTok link.")
        return
    ahead = queue.qsize()
    status = await update.message.reply_text(
        "⏳ Got it" + (f", queued (position {ahead + 1})" if ahead else ", working…"))
    await queue.put(Job(url, update.effective_chat.id, status.message_id, **opts))


async def on_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not await guard(update):
        return
    text = update.message.text or update.message.caption or ""
    words = text.lower().split()
    force = True if "frames" in words else False if "noframes" in words else None
    await enqueue(update, find_url(text), force_frames=force)


def command(**opts):
    async def handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if await guard(update):
            await enqueue(update, find_url(" ".join(ctx.args)), **opts)
    return handler


async def on_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if await guard(update):
        await update.message.reply_text(HELP)


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


class Progress:
    """Edits the status message from the worker thread, at most once every few seconds."""

    def __init__(self, app: Application, loop: asyncio.AbstractEventLoop, job: Job):
        self.app, self.loop, self.job = app, loop, job
        self.last_sent, self.last_text = 0.0, ""

    def __call__(self, text: str) -> None:
        now = time.monotonic()
        if text == self.last_text or now - self.last_sent < 3:
            return
        self.last_sent, self.last_text = now, text
        asyncio.run_coroutine_threadsafe(self._edit(text), self.loop)

    async def _edit(self, text: str) -> None:
        try:
            await self.app.bot.edit_message_text(text, self.job.chat_id, self.job.status_id)
        except (BadRequest, RetryAfter) as e:  # "message is not modified", flood control
            log.debug("progress edit skipped: %s", e)


async def worker(app: Application) -> None:
    loop = asyncio.get_running_loop()
    while True:
        job = await queue.get()
        try:
            result = await asyncio.to_thread(
                pipeline.run, job.url, Progress(app, loop, job), force_frames=job.force_frames,
                use_cache=job.use_cache, transcript_only=job.transcript_only)
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


async def post_init(app: Application) -> None:
    app.create_task(worker(app))
    log.info("bot ready; allowed users: %s", sorted(config.ALLOWED_USER_IDS) or "NONE (setup mode)")


def main() -> None:
    if not config.TELEGRAM_BOT_TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set (.env)")
    app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).post_init(post_init).build()
    app.add_handler(CommandHandler(["start", "help"], on_help))
    app.add_handler(CommandHandler("frames", command(force_frames=True)))
    app.add_handler(CommandHandler("noframes", command(force_frames=False)))
    app.add_handler(CommandHandler("again", command(use_cache=False)))
    app.add_handler(CommandHandler("transcript", command(transcript_only=True)))
    app.add_handler(MessageHandler((filters.TEXT | filters.CAPTION) & ~filters.COMMAND, on_message))
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
