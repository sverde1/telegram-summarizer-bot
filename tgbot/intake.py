"""Everything that turns a user's message into a job: links, uploaded files, shared files, book menus."""
import asyncio
import logging
import shutil
import time
from pathlib import Path

from telegram import Update
from telegram.ext import ContextTypes

from summarizer import config, db, links, memory, pipeline, transcribe
from summarizer.urls import UnsupportedURL, check as check_url, find_url
from tgbot import handlers, jobs, limits, menus, render, state, texts

log = logging.getLogger("bot")  # one logger name for the whole bot, as in the journal


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
        link = links.parse(url)
    except links.LinkError as e:
        await update.message.reply_text(str(e))
        return
    if link:  # a Google Drive / Dropbox file: a recording, or a document
        await _handle_link(update, url, link, **opts)
        return
    try:
        check_url(url)  # no network: a bad link is refused before it gets a queue slot or a request row
    except UnsupportedURL as e:
        await update.message.reply_text(f"⚠️ {e}")
        return
    uid = update.effective_user.id
    if refusal := limits.refusal(uid):
        await update.message.reply_text(refusal)
        return
    status = await update.message.reply_text(limits.queued_message())
    kind = _request_kind(opts.get("transcript_only", False), opts.get("use_cache", True))
    await jobs.start_job(uid, update.effective_chat.id, status.message_id, url, kind, **opts)


def _request_kind(transcript_only: bool, use_cache: bool) -> str:
    """The request's kind as /history and the limits record it: "transcript", "again" or "summary"."""
    return "transcript" if transcript_only else "again" if not use_cache else "summary"


TG_DOWNLOAD_LIMIT = 20 * 1024 ** 2  # the most a bot may download from Telegram (Bot API getFile)


DOC_EXTENSIONS = {".pdf", ".epub", ".docx", ".txt"}


CONVERT_EXTENSIONS = {".doc", ".mobi", ".azw", ".azw3", ".rtf", ".odt", ".fb2", ".djvu"}


DOC_FORMATS = "PDF, EPUB, DOCX or TXT"


AUDIO_EXTENSIONS = {".mp3", ".m4a", ".wav", ".ogg", ".oga", ".opus", ".flac", ".aac", ".wma", ".amr"}


VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".3gp", ".m4v", ".mpeg", ".mpg", ".wmv"}


MEDIA_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS  # recordings, as opposed to documents


def _seconds(value) -> float:
    """A Telegram duration in seconds (a timedelta with PTB_TIMEDELTA, else a number; None = 0)."""
    return value.total_seconds() if hasattr(value, "total_seconds") else float(value or 0)


def _media_label(msg) -> tuple[str, object]:
    """What a sent recording is called in replies and /history, and the Telegram file object."""
    if msg.voice:
        return "🎤 Voice message", msg.voice
    if msg.video_note:
        return "🎥 Video message", msg.video_note
    if msg.audio:
        a = msg.audio
        name = a.file_name or " - ".join(x for x in (a.performer, a.title) if x) or "Audio file"
        return f"🎵 {name}", a
    if msg.video:
        return f"🎬 {msg.video.file_name or 'Video'}", msg.video
    d = msg.document
    is_video = ((d.mime_type or "").startswith("video/")
                or Path(d.file_name or "").suffix.lower() in VIDEO_EXTENSIONS)
    return f"{'🎬' if is_video else '🎵'} {d.file_name or 'Recording'}", d


def _is_media_document(d) -> bool:
    """Whether a sent document is a recording (audio/video type or extension) rather than a book."""
    return ((d.mime_type or "").split("/")[0] in ("audio", "video")
            or Path(d.file_name or "").suffix.lower() in MEDIA_EXTENSIONS)


async def on_media(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles a voice message, audio, video or round video message (or such a file as a document).

    It's summarized right away, like a link (with the caption /transcript: the transcript instead).
    Everything that can be checked from Telegram's description is checked before anything is downloaded:
    the size (bots get at most 20 MB), the length and whether it could ever fit in memory.
    """
    if not await handlers.guard(update, ctx):
        return
    msg = update.message
    label, f = _media_label(msg)
    uid = update.effective_user.id
    if not f.file_size or f.file_size > TG_DOWNLOAD_LIMIT:
        await msg.reply_text(texts.MEDIA_TOO_BIG)
        return
    seconds = _seconds(getattr(f, "duration", None))
    if seconds > config.MAX_DURATION_MIN * 60:
        await msg.reply_text(f"⚠️ This recording is longer than {config.MAX_DURATION_MIN} min; skipping.")
        return
    if seconds and not memory.can_ever_fit(memory.whisper_needs(seconds, transcribe.is_loaded())):
        await msg.reply_text("🧠 This recording is too long to transcribe on this machine.")
        return
    if refusal := limits.refusal(uid):
        await msg.reply_text(refusal)
        return
    if seconds:
        label = f"{label} ({pipeline.fmt_duration(seconds)})" if msg.voice or msg.video_note else label
    upload_id = db.add_upload(uid, f.file_id, f.file_unique_id, label, f.file_size)
    transcript = (msg.caption or "").strip().lower().startswith("/transcript")
    status = await msg.reply_text(limits.queued_message())
    await jobs.start_job(uid, update.effective_chat.id, status.message_id, label,
                     _request_kind(transcript, True), upload_id=upload_id, job_kind=state.JobKind.MEDIA,
                     transcript_only=transcript)


async def on_document(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles an uploaded file: checks its type and size, then asks how to summarize it."""
    if update.message.document and _is_media_document(update.message.document):
        await on_media(update, ctx)  # a recording sent as a file, not a book
        return
    if not await handlers.guard(update, ctx):
        return
    d = update.message.document
    name = d.file_name or "document"
    ext = Path(name).suffix.lower()
    if ext in CONVERT_EXTENSIONS:
        await update.message.reply_text(
            f"⚠️ I can't read {ext} files. Convert it to PDF or EPUB and send it again.")
        return
    if ext not in DOC_EXTENSIONS:
        await update.message.reply_text(f"⚠️ I can summarize {DOC_FORMATS} files, and YouTube or TikTok links.")
        return
    if not d.file_size or d.file_size > TG_DOWNLOAD_LIMIT:
        await update.message.reply_text(texts.TOO_BIG)
        return
    upload_id = db.add_upload(update.effective_user.id, d.file_id, d.file_unique_id, name, d.file_size)
    await update.message.reply_text(f"📄 {name} ({render.fmt_size(d.file_size)})\nHow should I summarize it?",
                                    reply_markup=menus.book_menu(upload_id))


def _link_kind(name: str, content_type: str) -> str | None:
    """"media", "document" or None (can't tell) for a shared file, by its name, else its type."""
    ext = Path(name).suffix.lower()
    if ext in MEDIA_EXTENSIONS:
        return "media"
    if ext in DOC_EXTENSIONS or ext in CONVERT_EXTENSIONS:
        return "document"
    if content_type.split("/")[0] in ("audio", "video"):
        return "media"
    if content_type in ("application/pdf", "application/epub+zip", "text/plain") or "wordprocessingml" in content_type:
        return "document"
    return None


async def _handle_link(update: Update, url: str, link: links.FileLink, transcript_only: bool = False,
                       use_cache: bool = True) -> None:
    """A Google Drive / Dropbox link: a recording is summarized right away (like an upload), a document gets
    the book menu.

    When the link doesn't show the file's name (Drive), its first bytes are fetched to learn the name, type and
    size: after the access and limit checks, since anyone allowed could trigger it. A recording's size and the
    free disk space are checked before it's queued.
    """
    uid = update.effective_user.id
    kind = _link_kind(link.name, "")
    info = {"filename": link.name, "content_type": "", "size": None}
    if kind is None and Path(link.name).suffix:  # the link names a file of another kind: no need to look
        await update.message.reply_text("⚠️ I can't tell what kind of file this is. I can summarize audio and "
                                        f"video files, and {DOC_FORMATS} documents.")
        return
    if kind != "document":
        if refusal := limits.refusal(uid):
            await update.message.reply_text(refusal)
            return
        try:
            info = await asyncio.to_thread(links.peek, link)
        except links.LinkError as e:
            await update.message.reply_text(str(e))
            return
        kind = _link_kind(info["filename"], info["content_type"])
    if kind == "document":
        if transcript_only:
            await update.message.reply_text("That's a document; send the link without /transcript.")
            return
        await _offer_link(update, url, links.FileLink(link.service, link.download_url, info["filename"]))
        return
    if kind is None:
        await update.message.reply_text("⚠️ I can't tell what kind of file this is. I can summarize audio and "
                                        f"video files, and {DOC_FORMATS} documents.")
        return
    size = info["size"] or 0
    if size > config.MAX_MEDIA_LINK_MB * 1024 ** 2:
        await update.message.reply_text(f"⚠️ This file is larger than {config.MAX_MEDIA_LINK_MB} MB, the most I "
                                        "download.")
        return
    if size and shutil.disk_usage(config.DATA_DIR).free < 2 * size:
        await update.message.reply_text("⚠️ There isn't enough free disk space for this file right now.")
        log.warning("disk too full for a %d MB file", size // 1024 ** 2)
        return
    name = info["filename"] or f"{link.service} file"
    audio = Path(name).suffix.lower() in AUDIO_EXTENSIONS or info["content_type"].startswith("audio/")
    icon = "🎵" if audio else "🎬"
    label = f"{icon} {name}"
    upload_id = db.add_link_upload(uid, url, label)
    status = await update.message.reply_text(limits.queued_message())
    await jobs.start_job(uid, update.effective_chat.id, status.message_id, label,
                     _request_kind(transcript_only, use_cache), upload_id=upload_id, job_kind=state.JobKind.MEDIA,
                     transcript_only=transcript_only, use_cache=use_cache)


async def _offer_link(update: Update, url: str, link: links.FileLink) -> None:
    """A Google Drive / Dropbox link to a document: records it like an upload and asks how to summarize it.

    Nothing is downloaded yet: that happens in the job, once the user picks (and the limits allow it).
    """
    name = link.name or f"{link.service} file"
    ext = Path(link.name).suffix.lower()
    if ext in CONVERT_EXTENSIONS:
        await update.message.reply_text(
            f"⚠️ I can't read {ext} files. Convert it to PDF or EPUB and send it again.")
        return
    if link.name and ext not in DOC_EXTENSIONS:
        await update.message.reply_text(f"⚠️ I can summarize {DOC_FORMATS} files, and YouTube or TikTok links.")
        return
    upload_id = db.add_link_upload(update.effective_user.id, url, name)
    await update.message.reply_text(f"📄 {name} ({link.service})\nHow should I summarize it?",
                                    reply_markup=menus.book_menu(upload_id))


async def on_book_button(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the buttons under an upload and in its chapter list (`book:<upload id>:<action>[:<n>]`).

    Actions: whole / short / each / pick start a job; chapters / back switch the menu; pg:<n> pages the
    chapter list; ch:<n> summarizes one chapter. Callback data can be forged, so everything is re-checked:
    private chat, access, and that the upload is the user's own (or the user is an admin).
    """
    if not (checked := await handlers.button(update)):
        return
    q, uid, parts = checked
    upload = db.get_upload(int(parts[1])) if len(parts) >= 3 and parts[1].isdigit() else None
    if upload is None or not handlers.owns(uid, upload["user_id"]):
        await q.answer("This isn't available.")
        return
    action, arg = parts[2], (int(parts[3]) if len(parts) == 4 and parts[3].isdigit() else None)
    if action in ("chapters", "back"):
        await q.answer()
        await q.edit_message_reply_markup(menus.book_menu(upload["id"], chapters=action == "chapters"))
        return
    if action == "pg" and arg is not None:
        doc = db.get_document(upload["sha256"]) if upload["sha256"] else None
        if not doc or not doc["chapters"]:
            await q.answer("Please pick \"By chapter\" again.")
            return
        text, markup = menus.chapter_list(upload["id"], upload["name"], doc["chapters"], arg)
        await q.answer()
        await q.edit_message_text(text, reply_markup=markup)
        return
    if action not in (*menus.BOOK_MODES, "ch") or (action == "ch" and arg is None):
        await q.answer("This isn't available.")
        return
    if action == "pick":
        # This upload was read already: its chapter list is stored, so show it now instead of queueing a
        # job (which would wait behind whatever runs, e.g. this book's other summaries). Nothing is
        # summarized, so nothing counts; the chapter tapped next is the request. Only for this very upload:
        # another user's upload of the same file reads it first, which keeps that a fresh-looking run.
        doc = db.get_document(upload["sha256"]) if upload["sha256"] else None
        if doc and doc["status"] == "done" and doc["chapters"]:
            text, markup = menus.chapter_list(upload["id"], upload["name"], doc["chapters"], 0)
            await q.answer()
            await ctx.bot.send_message(q.message.chat.id, text, reply_markup=markup)
            return
    request_id = None
    if action == "ch":  # the first chapter picked from a list continues the list's request
        request_id = db.waiting_request(uid, upload["sha256"]) if upload["sha256"] else None
    if refusal := limits.refusal(uid, new_request=request_id is None):
        await q.answer(refusal[:200], show_alert=True)
        return
    await q.answer()
    status = await ctx.bot.send_message(q.message.chat.id, limits.queued_message())
    mode, kind = ("pick", "chapter") if action == "ch" else (action, menus.BOOK_MODES[action][0])
    if request_id:
        db.update_request(request_id, status="queued", kind=kind)
    await jobs.start_job(uid, q.message.chat.id, status.message_id, f"📄 {upload['name']}", kind,
                     request_id=request_id, upload_id=upload["id"], book_mode=mode, chapter=arg,
                     job_kind=state.JobKind.DOCUMENT)


ASK_WINDOW = 300  # seconds after a 💬 tap in which a plain text (without a link) is the question


async def on_message(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles plain messages: a follow-up question (see _question_for), else the first link in the text or
    media caption is summarized."""
    if not await handlers.guard(update, ctx):
        return
    if parent := _question_for(update, ctx):
        await ask(update, parent, update.message.text.strip())
        return
    text = update.message.text or update.message.caption or ""
    await enqueue(update, find_url(text))


def _question_for(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> int | None:
    """The summary request a message asks about, if it is a question, else None.

    A question is a text (never a caption or file) that replies to one of the bot's summary, answer or 💬 prompt
    messages (db.messages: any part of a long summary works, and it survives restarts), or a text without a
    link sent within ASK_WINDOW of tapping 💬 without replying. Checked here rather than in a handler of its
    own: Telegram updates go to the first matching handler only, so a separate reply handler would swallow
    every other reply (a link sent as a reply, say).
    """
    msg = update.message
    if not msg.text:
        return None
    if (r := msg.reply_to_message) is not None:
        if r.from_user is None or r.from_user.id != ctx.bot.id:
            return None
        return db.message_request(msg.chat_id, r.message_id)
    pending = state.asking.get(update.effective_user.id)
    if pending and time.monotonic() - pending[1] < ASK_WINDOW and not find_url(msg.text):
        return pending[0]
    return None


async def ask(update: Update, parent_id: int, question: str) -> None:
    """Queues a follow-up question about a summary, after the checks a question needs.

    Only the summary's owner (or an admin) may ask; the question is capped; one open question per summary;
    the question limit (admins exempt) and the queue limits apply. Questions don't count toward the daily
    request limit.
    """
    msg, uid = update.message, update.effective_user.id
    req = db.get_request(parent_id)
    if req is None or not handlers.owns(uid, req["user_id"]):
        await msg.reply_text("This isn't available.")
        return
    if not db.get_delivered(parent_id):
        await msg.reply_text(texts.TOO_OLD)
        return
    if len(question) > config.ASK_MAX_QUESTION:
        await msg.reply_text(f"⚠️ Please keep the question under {config.ASK_MAX_QUESTION} characters.")
        return
    if any(j.ask_of == parent_id and j.user_id == uid and not j.cancel_reason for j in state.jobs.values()):
        await msg.reply_text("💬 Still working on your last question about this; ask again when it's answered.")
        return
    used, limit, _, frees_in = limits.limit_status(uid, "ask")
    if limit and used >= limit:
        await msg.reply_text(f"⏳ You've asked {limit} questions today. You can ask more in about "
                             f"{render.fmt_until(frees_in or 0)}.")
        return
    if refusal := limits.refusal(uid, new_request=False):
        await msg.reply_text(refusal)
        return
    state.asking.pop(uid, None)
    status = await msg.reply_text(limits.queued_message())
    await jobs.start_job(uid, msg.chat_id, status.message_id, question, "ask", job_kind=state.JobKind.ASK,
                         ask_of=parent_id, question=question, reply_to=msg.message_id)


def command(**opts):
    """Makes a handler for a `/command [url]` that queues the link with the given job options.

    Args:
        **opts: Job options, e.g. `use_cache=False` (/again) or `transcript_only=True` (/transcript).

    Returns:
        An async command handler.
    """
    async def handler(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Queues the URL given as the command's argument, for allowed users; without one, the user's latest
        video link (so "/again" after a failure redoes it)."""
        if await handlers.guard(update, ctx):
            url = find_url(" ".join(ctx.args)) or db.last_video_link(update.effective_user.id)
            await enqueue(update, url, **opts)
    return handler
