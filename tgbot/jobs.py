"""The job queue's life cycle: queueing, the worker, cancelling, finishing, failing, admin notices."""
import asyncio
import logging
import time

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import BadRequest, TelegramError
from telegram.ext import Application

import access
from summarizer import config, db, documents, memory, pipeline, proc, stats, summarize, transcribe
from summarizer.results import JobResult
from summarizer.urls import UnsupportedURL
from tgbot import delivery, limits, menus, prefs, render, runners, sending, state, texts

log = logging.getLogger("bot")  # one logger name for the whole bot, as in the journal


async def report_cancel(app: Application, job: state.Job) -> None:
    """Records a job as cancelled and shows the reason on its status message."""
    reason = job.cancel_reason or texts.CANCELLED
    db.update_request(job.request_id, status="cancelled", error="cancelled: " + reason[:200])
    await fail(app, job, reason)


async def cancel_job(app: Application, job: state.Job, reason: str) -> bool:
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
    if job.cancel_reason or job.request_id not in state.jobs:
        return False
    job.cancel_reason = reason
    if job is state.running:
        proc.current_job_cancel.set()
        return True
    task = state.delayed.pop(job.request_id, None)
    if task:
        task.cancel()
    _unpark(job)
    end_job(job)
    await report_cancel(app, job)
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
    mine = [j for j in state.jobs.values() if j.user_id == uid and not j.cancel_reason]
    for job in mine:
        await cancel_job(app, job, reason)
    if mine:
        log.info("cancelled %d job(s) of user %s", len(mine), uid)
    return len(mine)


async def start_job(uid: int, chat_id: int, status_id: int, url: str, request_kind: str,
                     request_id: int | None = None, **opts) -> state.Job:
    """Logs the request (unless it continues one) and queues the job; the caller sent the status message.

    Args:
        uid: The requesting user.
        chat_id: Their chat.
        status_id: The status message the worker edits as the job progresses.
        url: The link, or the file name for documents.
        request_kind: The request's kind for /history and the limits (summary, again, transcript, book, ...).
        request_id: An existing request to continue, or None for a new one.
        **opts: Job fields (job_kind, use_cache, transcript_only, upload_id, book_mode, chapter, ...).
    """
    req = request_id or db.add_request(uid, url, request_kind)  # logged the moment it arrives
    b, m, is_default = prefs.current_llm(uid)
    # Users on the defaults pass None, so their jobs follow the default (and its later changes)
    # instead of pinning whatever the default resolves to right now.
    b, m = (None, None) if is_default else (b, m)
    state.user_jobs[uid] += 1
    job = state.Job(url, chat_id, status_id, queued_at=time.monotonic(), user_id=uid, request_id=req, backend=b,
              model=m, **opts)
    state.jobs[req] = job
    await state.queue.put(job)
    return job


MEMORY_RECHECK = 15  # seconds between memory re-checks for jobs set aside


def end_job(job: state.Job) -> None:
    """Forgets a job that's completely done (finished, failed, cancelled or expired): frees its user's slot.

    Idempotent: several paths can end the same job (e.g. a cancel while the worker is reporting it), and only
    the first one may free the slot, or the user's counter would go wrong.
    """
    if state.jobs.pop(job.request_id, None) is None:
        return
    state.user_jobs[job.user_id] -= 1
    if state.user_jobs[job.user_id] <= 0:
        del state.user_jobs[job.user_id]


def _unpark(job: state.Job) -> bool:
    """Takes a job off the memory-wait list; False if something else (a cancel) already did."""
    if job in state.waiting_for_memory:
        state.waiting_for_memory.remove(job)
        return True
    return False


async def set_aside(app: Application, job: state.Job, needed: int, duration: float = 0) -> None:
    """Parks a job that needs more memory than is free, and shows why, with a button to stop waiting."""
    if job.waiting_since is None:
        job.waiting_since = time.monotonic()
    job.memory_needed, job.audio_seconds = needed, duration
    state.waiting_for_memory.append(job)
    db.update_request(job.request_id, status="queued")
    what = "recording" if job.job_kind is state.JobKind.MEDIA else "video"
    text = (f"🧠 Not enough free memory to transcribe this {what} right now (needs about "
            f"{needed / memory.GB:.1f} GB). Waiting up to {config.WHISPER_RAM_WAIT_MIN} min; other requests go "
            "first meanwhile.")
    button = InlineKeyboardMarkup([[InlineKeyboardButton("✖️ Don't wait, cancel",
                                                         callback_data=f"cancel:{job.request_id}")]])
    try:
        await app.bot.edit_message_text(text, job.chat_id, job.status_id, reply_markup=button)
    except TelegramError as e:
        log.debug("couldn't show the memory wait: %s", e)


def _memory_needed_now(job: state.Job) -> int:
    """A waiting job's memory need, re-estimated: the model may have been loaded by another job since (it's
    then part of the bot's memory already and mustn't be counted again)."""
    if not job.audio_seconds:
        return job.memory_needed
    return memory.whisper_needs(job.audio_seconds, transcribe.is_loaded())


async def next_job(app: Application) -> tuple[state.Job | None, bool]:
    """Picks the next job: a parked one whose memory now fits, else the next one from the queue.

    Parked jobs past their wait limit are failed here. While jobs are parked, waiting on the queue times out
    every MEMORY_RECHECK seconds so they're re-checked even when nothing new arrives.

    Returns:
        (job or None if nothing to do yet, whether it came from the queue).
    """
    now = time.monotonic()
    # Iterate over a snapshot, and re-check membership: the awaits below let a cancel take jobs off the list.
    for job in list(state.waiting_for_memory):
        if job not in state.waiting_for_memory:
            continue
        if job.cancel_reason:  # cancelled while parked: already reported
            _unpark(job)
            end_job(job)
        elif now - job.waiting_since > config.WHISPER_RAM_WAIT_MIN * 60:
            _unpark(job)
            end_job(job)  # bookkeeping before the await, so a concurrent cancel finds nothing to undo
            db.update_request(job.request_id, status="failed", error="waited too long for memory")
            await fail(app, job, f"🧠 Still not enough free memory after {config.WHISPER_RAM_WAIT_MIN} min. "
                                  "Please try again later.")
        elif memory.fits_now(_memory_needed_now(job)):
            _unpark(job)
            return job, False
    try:
        timeout = MEMORY_RECHECK if state.waiting_for_memory else None
        return await asyncio.wait_for(state.queue.get(), timeout), True
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
    loop = asyncio.get_running_loop()
    while True:
        job, from_queue, parked = None, False, False
        # Everything, job selection included, is inside the safety net: a database hiccup or an unexpected
        # state while picking a job must not end the loop. Only Exception: shutdown's CancelledError must pass.
        try:
            job, from_queue = await next_job(app)
            if job is None:
                continue
            if access.state(job.user_id) not in ("admin", "allowed") and not job.cancel_reason:
                # Access removed while queued (e.g. a path that didn't go through cancel_user_jobs).
                job.cancel_reason = texts.ACCESS_REMOVED
                db.update_request(job.request_id, status="cancelled", error="cancelled: access removed")
                await fail(app, job, texts.ACCESS_REMOVED)
            if not job.cancel_reason:  # cancelled while queued: already reported, just skip it
                parked = await _run_job(app, loop, job)
        except Exception:
            log.exception("worker: job %s failed", job.request_id if job else "(selecting the next job)")
            await asyncio.sleep(1)  # don't spin if the failure repeats (e.g. the database is locked)
        finally:
            state.running = None
            if job is not None and not parked:
                end_job(job)
            if from_queue:
                state.queue.task_done()


async def _run_job(app: Application, loop: asyncio.AbstractEventLoop, job: state.Job) -> bool:
    """Runs one job: pipeline, reply, request bookkeeping. Failures are reported to the user and recorded.

    Returns:
        True if the job was set aside to wait for memory (it isn't done yet), else False.

    Args:
        app: The running application.
        loop: The event loop (for thread-safe status edits from the pipeline thread).
        job: The job to run.
    """
    waited = time.monotonic() - job.queued_at
    job.started_at = time.monotonic()
    proc.current_job_cancel.clear()
    state.running = job
    try:
        progress = sending.Progress(app, loop, job)
        try:
            result = await runners.RUNNERS[job.job_kind](app, job, progress)
        finally:
            # Before replying or deleting the status message: a late status edit must not land
            # after the final result.
            await progress.close()
        if job.cancel_reason:  # cancelled while the pipeline ran but finished before noticing (e.g. cached)
            raise proc.ProcCancelled("cancelled")
        if result.hold or (result.cached and result.replay_steps):  # a first-time requester (see plan_replay)
            # Replayed in a separate task so the queue keeps moving meanwhile; the task ends the job.
            state.delayed[job.request_id] = asyncio.create_task(_deliver_later(app, job, result, waited))
            return True
        await _finish(app, job, result, waited)
    except Exception as e:
        # A cancel wins over whatever error it caused: e.g. a download killed by the cancel (or by systemd
        # at shutdown) must be reported as the cancel, not as "couldn't load this video".
        if job.cancel_reason or isinstance(e, proc.ProcCancelled):
            await report_cancel(app, job)
            return False
        if isinstance(e, documents.NeedsOcr):
            await _ask_ocr(app, job, e)
            return False
        if isinstance(e, memory.NeedsMemory):
            await set_aside(app, job, e.needed, e.duration)
            return True
        if isinstance(e, sending.UserBlockedBot):
            db.update_request(job.request_id, status="failed", error="user blocked the bot")
            log.info("job %s: user %s blocked the bot", job.request_id, job.user_id)
        elif isinstance(e, (pipeline.PipelineError, UnsupportedURL)):
            # Expected failures: the message is written for the user; the technical detail is for the admin.
            detail = getattr(e, "detail", None)
            db.update_request(job.request_id, status="failed", error=(detail or str(e))[:500])
            if detail:
                log.warning("job %s failed: %s (%s)", job.request_id, e, detail)
            await fail(app, job, str(e), detail if access.is_admin(job.user_id) else None)
            if isinstance(e, pipeline.Blocked):
                await _notify_admins_of_block(app, e)
        else:  # report, don't crash the worker
            log.exception("job failed: %s", job.url)
            detail = f"{type(e).__name__}: {e}"
            db.update_request(job.request_id, status="failed", error=f"unexpected: {detail}"[:500])
            if access.is_admin(job.user_id):
                await fail(app, job, texts.INTERNAL_ERROR, detail)
            else:
                await fail(app, job, texts.INTERNAL_ERROR_NOTIFIED)
                await _notify_admins_of_error(app, job, e)
    return False


async def _ask_ocr(app: Application, job: state.Job, e: documents.NeedsOcr) -> None:
    """A scanned document: asks the user to confirm OCR (with the time it takes), or explains why it can't run.

    The request waits ("waiting") with its job remembered in ocr_holds; the Start button continues it, so it
    counts once toward the daily limit. Over OCR_MAX_PAGES the user can only accept or ask the admins.
    """
    db.save_ocr_hold(job.request_id, job.upload_id, job.book_mode, job.chapter, e.pages, e.seconds, e.language)
    hold = db.get_ocr_hold(job.request_id)
    if refusal := (None if access.is_admin(job.user_id) else limits.ocr_refusal(job.user_id)):
        db.update_request(job.request_id, status="failed", error="OCR limit reached")
        await fail(app, job, refusal)
        return
    over_cap = e.pages > config.OCR_MAX_PAGES and not hold["approved"] and not access.is_admin(job.user_id)
    if over_cap:
        text = (f"⚠️ This scan has {e.pages} pages that need text recognition (OCR); I read up to "
                f"{config.OCR_MAX_PAGES} pages. Reading all {e.pages} would take about {render.fmt_eta(e.seconds)}.")
    else:
        text = (f"🔍 This is a scanned document: {e.pages} pages need text recognition (OCR), about "
                f"{render.fmt_eta(e.seconds)}, plus the summary. Start?")
    db.update_request(job.request_id, status="waiting")
    await app.bot.edit_message_text(text, chat_id=job.chat_id, message_id=job.status_id,
                                    reply_markup=menus.ocr_buttons(job.request_id, over_cap))


async def _finish(app: Application, job: state.Job, result: JobResult, waited: float) -> None:
    """Hands a finished job's result to the user and closes its request.

    A document in "pick" mode shows its chapter list instead (the request stays open for the pick). Otherwise
    the result is delivered, the request marked done and the status message deleted.

    Raises:
        UserBlockedBot: The user can't be reached.
        TelegramError: Telegram refused the result.
    """
    if isinstance(result, documents.DocResult) and result.kind == "pick":
        await delivery.show_pick_list(app, job, result)
        return
    await delivery.deliver(app, job, result, waited)
    db.update_request(job.request_id, status="done", cached=int(result.cached))
    # Only fresh summaries teach the wait estimate: cache hits (~1 s) and transcripts would drag the average
    # far below what a queued link really waits for. The worker's time, without any privacy pause.
    if not result.cached and not job.transcript_only and (total := result.work_seconds()):
        stats.record("job", total)
    try:
        await app.bot.delete_message(job.chat_id, job.status_id)
    except TelegramError:
        pass  # the user deleted it already, or can't be reached: the result is delivered either way


async def _deliver_later(app: Application, job: state.Job, result: JobResult, waited: float) -> None:
    """Finishes a first-time requester's job off the worker, so its pacing doesn't hold up anyone else.

    Either waits out the time still owed after real work on reused data (result.hold: status text and
    seconds), or replays a fully cached answer like a fresh run (result.replay_steps, already scaled); then
    delivers (or shows the chapter list). Stops if the job is cancelled meanwhile. Always ends the job.
    """
    try:
        progress = sending.Progress(app, asyncio.get_running_loop(), job)
        head = result.head()
        llm = result.llm_label() or summarize.llm_label(job.backend, job.model or "")
        stages = result.hold or [(f"{head}\n{pipeline.replay_stage(name, llm)}", sec)
                                 for name, sec in result.replay_steps]
        remaining = sum(sec for _, sec in stages)
        try:
            for text, sec in stages:
                if job.cancel_reason:
                    return
                progress(text, remaining)
                await asyncio.sleep(sec)
                remaining -= sec
        finally:
            await progress.close()
        if job.cancel_reason:
            return
        await _finish(app, job, result, waited)
    except proc.ProcCancelled:
        await report_cancel(app, job)  # cancelled between parts of the summary
    except sending.UserBlockedBot:
        db.update_request(job.request_id, status="failed", error="user blocked the bot")
    except Exception:
        log.exception("replaying cached result for job %s failed", job.request_id)
    finally:
        state.delayed.pop(job.request_id, None)
        end_job(job)


ADMIN_ERROR_NOTICE_EVERY = 600  # seconds: one notice per error type, so a recurring bug can't flood the admins


async def _notify_admins_of_error(app: Application, job: state.Job, e: Exception) -> None:
    """Tells the admins about an unexpected error in another user's job (at most once per type per 10 min)."""
    kind = type(e).__name__
    now = time.monotonic()
    if now - state.admin_error_noticed.get(kind, -ADMIN_ERROR_NOTICE_EVERY) < ADMIN_ERROR_NOTICE_EVERY:
        return
    state.admin_error_noticed[kind] = now
    who = access.label(job.user_id, db.get_user(job.user_id))
    text = f"⚠️ Unexpected error in a job for {who}\nLink: {job.url}\nDetails: {kind}: {str(e)[:500]}"
    for admin in access.ADMINS:
        try:
            await app.bot.send_message(admin, text, disable_web_page_preview=True)
        except TelegramError as err:
            log.warning("couldn't notify admin %s: %s", admin, err)


BLOCK_NOTICE_EVERY = 6 * 3600  # seconds: a block lasts hours; one notice per platform in that time is enough


async def _notify_admins_of_block(app: Application, e: "pipeline.Blocked") -> None:
    """Tells the admins that YouTube/TikTok is blocking the server (at most once per platform per 6 h).

    They may need to wait it out, update yt-dlp, or set up cookies/a proxy.
    """
    now = time.monotonic()
    if now - state.block_noticed.get(e.platform, -BLOCK_NOTICE_EVERY) < BLOCK_NOTICE_EVERY:
        return
    state.block_noticed[e.platform] = now
    text = (f"🚫 {e.platform} is blocking downloads from this server. Users are being told to try later.\n"
            f"Raw error: {(e.detail or '')[:500]}\nOptions: wait, update yt-dlp, or add cookies/a proxy.")
    for admin in access.ADMINS:
        try:
            await app.bot.send_message(admin, text, disable_web_page_preview=True)
        except TelegramError as err:
            log.warning("couldn't notify admin %s: %s", admin, err)


async def fail(app: Application, job: state.Job, msg: str, detail: str | None = None) -> None:
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
