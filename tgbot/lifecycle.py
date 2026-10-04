"""Start-up, shutdown, the per-user command menus, update notices and the error handler."""
import asyncio
import html
import logging
import time

from telegram import BotCommand, BotCommandScopeChat, BotCommandScopeDefault
from telegram.constants import ParseMode
from telegram.error import BadRequest, Forbidden, TelegramError
from telegram.ext import Application, ContextTypes

import access
from summarizer import config, db, media, ocr, pipeline, stats, tts, updates
from tgbot import jobs, state, texts

log = logging.getLogger("bot")  # one logger name for the whole bot, as in the journal


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
    BotCommand("units", "Units: metric/imperial, °C/°F"),
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


async def _check_pot(app: Application) -> None:
    """Checks the PO-token setup (YouTube's bot check) once; tells the admins if it's broken, not if absent.

    A setup that was never installed is the owner's choice: only logged. A broken one (half installed,
    mismatched versions, the sandbox failing) is told once per problem, remembered across restarts.
    """
    why = await asyncio.to_thread(media.check_pot)
    if not why:
        log.info("PO tokens: on (bgutil %s, Node in the sandbox)", media._pot.get("version"))
        return
    log.info("PO tokens: off (%s)", why)
    if why.startswith("not installed") or stats.recall("pot_problem") == why:
        return
    stats.remember("pot_problem", why)
    for admin in access.ADMINS:
        try:
            await app.bot.send_message(admin, f"⚠️ YouTube PO tokens are off: {why}")
        except TelegramError:
            pass


async def _prepare_tts() -> None:
    """Downloads Kokoro's model if missing (off the event loop) and checks voice messages can be made; logs
    what's missing for the admins. Until it's ready, summaries get no 🔊 button."""
    try:
        await asyncio.to_thread(tts.download_models)
    except Exception as e:  # noqa: BLE001  (logged; the bot works without voice messages)
        log.warning("couldn't download the text-to-speech model: %s", e)
    if found := await asyncio.to_thread(tts.refresh):
        log.warning("voice messages (🔊) are off: %s", "; ".join(found))
    else:
        log.info("voice messages (🔊) are ready")


async def post_init(app: Application) -> None:
    """Startup: registers the command menus and starts the worker and the update checker.

    Menus are re-synced for every admin and allowed user on each start, so the menus match the database
    and the code's command lists even after either changed while the bot was down.

    Args:
        app: The application being started.
    """
    if tts.language():  # voice messages wanted: get the model in the background (338 MB, once)
        app.bot_data["tts_setup"] = asyncio.create_task(_prepare_tts())
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
    await _check_pot(app)
    app.bot_data["workers"] = [asyncio.create_task(jobs.worker(app)) for _ in range(config.WORKERS)]
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


SHUTDOWN_WAIT = 30  # seconds to let the running job stop and report (systemd's TimeoutStopSec is 60)


async def post_stop(app: Application) -> None:
    """Shutdown: tells everyone still waiting that the bot stopped, stops their work, then the tasks.

    PTB calls this after polling stopped but before the bot's connection closes, so messages can still be
    sent. Every unfinished job is cancelled with STOPPED (queued, waiting for memory or replaying ones are
    reported at once; running ones have their programs killed and their workers report them). The running
    jobs get SHUTDOWN_WAIT seconds to do so (an API call or a model load can't be interrupted sooner); any
    that don't make it are reported from here. Only then are the workers and update checker cancelled (without
    that, a worker blocked in `queue.get()` was destroyed while pending: "Event loop is closed" errors).

    Args:
        app: The application being stopped.
    """
    running = set(state.running)
    for job in list(state.jobs.values()):
        await jobs.cancel_job(app, job, texts.STOPPED)
    deadline = time.monotonic() + SHUTDOWN_WAIT
    while running & state.running and time.monotonic() < deadline:
        await asyncio.sleep(0.2)
    for job in running:
        if job.request_id in state.jobs:  # its worker didn't get to report it in time
            jobs.end_job(job)
            await jobs.report_cancel(app, job)
    tasks = [*app.bot_data.get("workers", []), app.bot_data.get("update_checker"), app.bot_data.get("tts_setup")]
    for task in filter(None, tasks):
        task.cancel()


async def on_error(update: object, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Logs exceptions raised by handlers (python-telegram-bot's error-handler hook).

    Args:
        update: The update being handled, if any.
        ctx: Handler context; `ctx.error` is the exception.
    """
    log.error("update handling failed", exc_info=ctx.error)
