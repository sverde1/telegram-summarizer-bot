"""Sending a finished result: summary messages, transcript files, documents, voice messages."""
import dataclasses
import html
import io
import logging
import re
import shutil

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import Application

import access
from summarizer import db, documents, pipeline, tts, units
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
    if isinstance(result, documents.DocResult):
        reveal = access.is_admin(job.user_id) or db.user_saw_video(job.user_id, "document", result.video_id,
                                                                    job.request_id)
        units_ = db.get_user_units(job.user_id)
        chunks = render.render_document(result, waited, reveal, units_)
        # Under a whole-book summary: the chapter options, for a reader who wants more detail.
        more = (menus.book_menu(job.upload_id, chapters=True, back=False)
                if result.kind == "book" and len(result.doc.get("chapters") or []) > 1 else None)
        more = _with_listen(more, job, *tts.document_text(_spoken_document(result, units_)))
        for k, chunk in enumerate(chunks, 1):
            await sending.send_with_retry(lambda chunk=chunk, k=k: bot_.send_message(
                job.chat_id, chunk, parse_mode=ParseMode.HTML, disable_web_page_preview=True,
                reply_markup=more if k == len(chunks) else None), job)
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
    listen = _with_listen(None, job, *tts.video_text(_spoken_summary(result.summary or {}, units_),
                                                     (result.meta or {}).get("title", "")))
    for k, chunk in enumerate(chunks, 1):
        await sending.send_with_retry(lambda chunk=chunk, k=k: bot_.send_message(
            job.chat_id, chunk, parse_mode=ParseMode.HTML, disable_web_page_preview=True,
            reply_markup=listen if k == len(chunks) else None), job)


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


def _with_listen(markup: InlineKeyboardMarkup | None, job: state.Job, title: str,
                 text: str) -> InlineKeyboardMarkup | None:
    """Adds the 🔊 Listen button to a summary's last message, storing first what it will read.

    Stored before the message is sent, so even an immediate tap finds it; it's exactly what the user reads.
    No button when voice messages aren't available or there's nothing to read.
    """
    lang = tts.language()
    if not tts.available() or not lang or not text.strip():
        return markup
    db.save_spoken(job.request_id, title, text, *lang)
    rows = list(markup.inline_keyboard) if markup else []
    return InlineKeyboardMarkup(rows + [[InlineKeyboardButton("🔊 Listen", callback_data=f"voice:{job.request_id}")]])


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
