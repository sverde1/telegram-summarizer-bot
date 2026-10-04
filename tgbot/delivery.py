"""Sending a finished result: summary messages, transcript files, documents, voice messages."""
import dataclasses
import html
import io
import logging
import re
import shutil

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ReplyParameters
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import Application

import access
from summarizer import db, documents, followup, pipeline, tts, units
from summarizer.results import JobResult
from tgbot import menus, render, runners, sending, state

log = logging.getLogger("bot")  # one logger name for the whole bot, as in the journal


async def show_pick_list(app: Application, job: state.Job, result: documents.DocResult) -> None:
    """The status message becomes the chapter list (kept, not deleted); the first chapter picked from it
    continues this request."""
    text, markup = menus.chapter_list(job.upload_id, result.name, result.doc["chapters"], 0)
    await app.bot.edit_message_text(text, chat_id=job.chat_id, message_id=job.status_id, reply_markup=markup)
    db.update_request(job.request_id, status="waiting")


async def deliver(app: Application, job: state.Job, result: JobResult, waited: float) -> None:
    """Sends a finished job's result: the transcript file, or the summary message(s).

    Raises:
        UserBlockedBot: The user can't be reached.
        TelegramError: Telegram refused the message.
    """
    bot_ = app.bot
    if isinstance(result, tts.VoiceResult):
        await _send_voice(app, job, result)
        return
    if isinstance(result, followup.AskResult):
        await _send_answer(app, job, result)
        return
    if isinstance(result, documents.DocResult):
        reveal = access.is_admin(job.user_id) or db.user_saw_video(job.user_id, "document", result.video_id,
                                                                    job.request_id)
        units_ = db.get_user_units(job.user_id)
        chunks = render.render_document(result, waited, reveal, units_)
        # Under a whole-book summary: the chapter options, for a reader who wants more detail.
        more = (menus.book_menu(job.upload_id, chapters=True, back=False)
                if result.kind == "book" and len(result.doc.get("chapters") or []) > 1 else None)
        more = _with_actions(more, job, result, units_)
        await _send_chunks(app, job, chunks, more)
        return
    if job.transcript_only:
        if not result.transcript:
            await sending.send_with_retry(
                lambda: bot_.send_message(job.chat_id, "No transcript available for this video."),
                                   job)
        else:
            data = result.transcript.encode()
            # A long transcript is a sizeable upload; PTB's default 5 s write timeout is too short for it.
            await sending.send_with_retry(lambda: bot_.send_document(
                job.chat_id, io.BytesIO(data), filename=transcript_name(result),
                caption=f"Transcript ({result.transcript_source})", write_timeout=60), job)
        return
    # Only admins, or the user who asked for this video before, may learn it was cached: otherwise it
    # would reveal what other users submit.
    reveal = access.is_admin(job.user_id) or db.user_saw_video(
        job.user_id, result.platform, result.video_id, job.request_id)
    units_ = db.get_user_units(job.user_id)
    chunks = render.render(result, waited, reveal_cache=reveal, units_=units_)
    await _send_chunks(app, job, chunks, _with_actions(None, job, result, units_))


async def _send_chunks(app: Application, job: state.Job, chunks: list[str],
                       markup: InlineKeyboardMarkup | None, parent_id: int | None = None,
                       reply_to: int = 0) -> None:
    """Sends a summary's (or an answer's) messages, the buttons under the last one, and remembers every
    message's id: a reply to any part of it is a question about that summary.

    Args:
        parent_id: The summary request replies ask about; this job's own request by default.
        reply_to: A message the first one replies to (the question); sent anyway if it was deleted.
    """
    ids = []
    for k, chunk in enumerate(chunks, 1):
        reply = ReplyParameters(reply_to, allow_sending_without_reply=True) if reply_to and k == 1 else None
        sent = await sending.send_with_retry(lambda chunk=chunk, k=k, reply=reply: app.bot.send_message(
            job.chat_id, chunk, parse_mode=ParseMode.HTML, disable_web_page_preview=True, reply_parameters=reply,
            reply_markup=markup if k == len(chunks) else None), job)
        if sent is not None:
            ids.append(sent.message_id)
    db.save_messages(job.chat_id, ids, parent_id or job.request_id)


def ask_button(parent_id: int) -> InlineKeyboardButton:
    """The 💬 Ask button for a summary (also under each answer: it asks about the same summary)."""
    return InlineKeyboardButton("💬 Ask", callback_data=f"ask:{parent_id}")


async def _send_answer(app: Application, job: state.Job, result: followup.AskResult) -> None:
    """Sends an answer as a reply to its question, with 💬 to ask on, and records it for the next questions.

    The answer is stored before it's sent (like a summary), so a quick follow-up already sees it.
    """
    db.save_delivered(job.request_id, "ask", result.question, result.answer, parent_id=result.parent_id)
    chunks = render.answer(result.answer, db.get_user_units(job.user_id))
    await _send_chunks(app, job, chunks, InlineKeyboardMarkup([[ask_button(result.parent_id)]]),
                       parent_id=result.parent_id, reply_to=job.reply_to)


def transcript_name(result: pipeline.Result) -> str:
    """The transcript file's name: the video id, or for a file someone sent, its (sanitised) label."""
    if result.platform != "file":
        return f"{result.video_id}.txt"
    name = re.sub(r"[^\w .-]+", "", result.url).strip(" .") or "recording"
    return f"{name[:80]} transcript.txt"


def _spoken_summary(summary: dict, units_: tuple[str, str]) -> dict:
    """A copy of a video summary as it's read aloud: the reader's units, written as words. Titles stay."""
    return {**summary, **{k: units.convert(summary.get(k, ""), *units_, spoken=True)
                          for k in ("clickbait_answer", "summary")}}


def _spoken_document(result: documents.DocResult, units_: tuple[str, str]) -> documents.DocResult:
    """A copy of a document result as it's read aloud (see _spoken_summary); the original isn't changed."""
    book = ({**result.book, "summary": units.convert(result.book.get("summary", ""), *units_, spoken=True)}
            if result.book else None)
    chapters = [(i, t, units.convert(s, *units_, spoken=True)) for i, t, s in result.chapters]
    return dataclasses.replace(result, book=book, chapters=chapters)


def _delivered(job: state.Job, result: JobResult) -> tuple[str, str, str, str, str]:
    """What a summary showed, as plain text before unit conversion (what 💬 and 📄 work from).

    Returns:
        (kind video|file|document, title, text, backend, model).
    """
    if isinstance(result, documents.DocResult):
        if result.kind == "book":
            b = result.book or {}
            author = f"Author: {b['author']}\n\n" if b.get("author") else ""
            title, text = b.get("title") or result.name, author + b.get("summary", "")
        else:
            title = result.name
            text = "\n\n".join(f"{t}\n{s}" for _, t, s in result.chapters)
        return "document", title, text, result.backend, result.model
    s = result.summary or {}
    stats_ = s.get("_stats") or {}
    # A recording's title is the summary's or its label: never the neutral title of the shared videos row.
    title = s.get("title") or (job.url if result.platform == "file" else (result.meta or {}).get("title", ""))
    text = s.get("summary", "")
    if result.platform != "file" and s.get("is_clickbait") and s.get("clickbait_answer"):
        text = f"Clickbait answer: {s['clickbait_answer']}\n\n{text}"
    return ("file" if result.platform == "file" else "video", title, text, stats_.get("backend"),
            stats_.get("model"))


def _with_actions(markup: InlineKeyboardMarkup | None, job: state.Job, result: JobResult,
                  units_: tuple[str, str]) -> InlineKeyboardMarkup | None:
    """Adds the summary's buttons under its last message, storing first what they work from.

    Stored before the message is sent, so even an immediate tap finds it; it's exactly what the user reads.
    Called only from deliver(), so only for summaries that were really delivered (after any pacing).
    🔊 only when voice messages are available and there's something to read.
    """
    kind, title, text, backend, model = _delivered(job, result)
    db.save_delivered(job.request_id, kind, title, text, backend, model)
    if isinstance(result, documents.DocResult):
        spoken_title, spoken = tts.document_text(_spoken_document(result, units_))
    else:
        spoken_title, spoken = tts.video_text(_spoken_summary(result.summary or {}, units_),
                                              (result.meta or {}).get("title", ""))
    buttons = []
    lang = tts.language()
    if tts.available() and lang and spoken.strip():
        db.save_spoken(job.request_id, spoken_title, spoken, *lang)
        buttons.append(InlineKeyboardButton("🔊 Listen", callback_data=f"voice:{job.request_id}"))
    buttons += [ask_button(job.request_id), InlineKeyboardButton("📄 Download", callback_data=f"md:{job.request_id}")]
    rows = list(markup.inline_keyboard) if markup else []
    return InlineKeyboardMarkup(rows + [buttons])


async def _send_voice(app: Application, job: state.Job, result: tts.VoiceResult) -> None:
    """Sends the voice message: the made file, or the reused one by Telegram's id (made again if Telegram no
    longer knows that id). A made one's id is remembered for VOICE_CACHE_DAYS."""
    caption = f"🔊 {html.escape(result.title[:200])}"
    try:
        if result.file_id:
            try:
                await sending.send_with_retry(lambda: app.bot.send_voice(job.chat_id, result.file_id, caption=caption,
                                                                  parse_mode=ParseMode.HTML), job)
                return
            except BadRequest as e:  # an expired or unknown id: make it again
                log.info("voice %s no longer usable (%s); making it again", result.key[:12], e)
                db.delete_voice(result.key)
                result.ogg, result.cached = await runners.make_voice(job, result), False
                db.update_request(job.request_id, cached=0)
        data = result.ogg.read_bytes()
        sent = await sending.send_with_retry(lambda: app.bot.send_voice(
            job.chat_id, io.BytesIO(data), caption=caption, parse_mode=ParseMode.HTML, write_timeout=120), job)
        if sent and sent.voice:
            duration = sent.voice.duration  # a timedelta with PTB_TIMEDELTA (see the top of this file)
            seconds = int(duration.total_seconds()) if hasattr(duration, "total_seconds") else duration
            db.save_voice(result.key, sent.voice.file_id, seconds)
    finally:
        if result.ogg:
            shutil.rmtree(result.ogg.parent, ignore_errors=True)
