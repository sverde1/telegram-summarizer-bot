"""One runner per job kind (RUNNERS): gets the input, runs the blocking work in a thread."""
import asyncio
import shutil
import time
from collections.abc import Callable
from pathlib import Path

from telegram.error import BadRequest, TelegramError
from telegram.ext import Application

import access
from summarizer import config, cpu, db, documents, followup, group, links, media, memory, pipeline, proc, tts
from tgbot import limits, render, sending, state, texts


async def _download_link(upload: dict, path: Path, head: str, progress: sending.Progress, max_mb: int | None = None,
                         rename: bool = True) -> dict:
    """Downloads a Drive/Dropbox-linked document (in a thread: it can take minutes), showing the progress.

    Returns:
        The upload row, with the file's real name once the service revealed it.

    Raises:
        documents.DocumentError: Not shared publicly, too large, or the download failed.
    """
    link = links.parse(upload["source_url"])  # parsed again: only the bot-built URL is ever fetched

    def shown(done: int, total: int | None) -> None:
        """Download progress in the status message."""
        of = f" of {render.fmt_size(total)}" if total else ""
        progress(f"{head}\n📥 Downloading the file… {render.fmt_size(done)}{of}", None)

    try:
        name = await asyncio.to_thread(links.download, link, path, shown, max_mb)
    except links.LinkError as e:
        raise documents.DocumentError(str(e), e.detail)
    if rename and name and name != upload["name"]:  # a recording keeps its label (set when the link came in)
        db.set_upload_name(upload["id"], name)
        upload = db.get_upload(upload["id"])
    return upload


async def _fetch_telegram_file(app: Application, file_id: str, path: Path, too_big: str, error) -> None:
    """Downloads a file the user sent to Telegram (bots may download only up to 20 MB).

    Args:
        app: The running application.
        file_id: Telegram's file id.
        path: Where to save it.
        too_big: The user message when Telegram refuses the size.
        error: The exception class to raise, called with (user message, technical detail).

    Raises:
        error: Too big, or the download failed.
    """
    try:
        tg_file = await app.bot.get_file(file_id)
        await tg_file.download_to_drive(path)
    except BadRequest as e:
        raise error(too_big if "too big" in str(e).lower() else texts.DOWNLOAD_FAILED, str(e))
    except TelegramError as e:
        raise error(texts.DOWNLOAD_FAILED, str(e))


async def _run_video(app: Application, job: state.Job, progress: sending.Progress) -> pipeline.Result:
    """Runs a YouTube/TikTok link job (in a thread: downloads, Whisper and LLM programs block).

    Raises:
        pipeline.PipelineError, UnsupportedURL: The link can't be summarized (message for the user).
        memory.NeedsMemory: Not enough free memory right now (the job is set aside).
    """
    return await asyncio.to_thread(
        pipeline.run, job.url, progress, use_cache=job.use_cache, request_id=job.request_id,
        backend=job.backend, model=job.model, transcript_only=job.transcript_only,
        again_limit_user=None if access.is_admin(job.user_id) else job.user_id,
        hide_cache_from=None if access.is_admin(job.user_id) else job.user_id)


async def _run_media(app: Application, job: state.Job, progress: sending.Progress) -> pipeline.Result:
    """Runs a recording job: gets the file (Telegram or a share link), reads it in the sandbox, summarizes it.

    The job directory is the bot's: a job that has to wait for memory keeps it, and resumes from the file
    already there instead of downloading it again.

    Raises:
        pipeline.PipelineError: The file can't be downloaded, read or summarized.
        memory.NeedsMemory: Not enough free memory right now (the job is set aside).
    """
    upload = db.get_upload(job.upload_id)
    if upload is None:
        raise pipeline.PipelineError("⚠️ Please send the file again.", detail=f"upload {job.upload_id} missing")
    workdir = config.DATA_DIR / "work" / f"media_{job.request_id}"
    sources = [p for p in workdir.glob("*") if p.stem in ("vid", "audio") and p.suffix != ".part"]
    keep = False
    try:
        if not sources:
            shutil.rmtree(workdir, ignore_errors=True)
            workdir.mkdir(parents=True)
            head = upload["name"][:80]
            progress(f"{head}\n📥 Getting the file…", None)
            path = workdir / "download"
            if upload["source_url"]:
                await _download_link(upload, path, head, progress, config.MAX_MEDIA_LINK_MB, rename=False)
            else:
                await _fetch_telegram_file(app, upload["file_id"], path, texts.MEDIA_TOO_BIG, pipeline.PipelineError)
            if job.cancel_reason:
                raise proc.ProcCancelled("cancelled")
            try:
                info = await asyncio.to_thread(media.probe_file, path, workdir)
            except media.MediaError as e:
                raise pipeline.PipelineError(texts.UNREADABLE_MEDIA, detail=str(e))
            if not info["has_audio"] and not info["has_video"]:
                raise pipeline.PipelineError(texts.UNREADABLE_MEDIA, detail="no audio or video stream")
            # The pipeline's conventions: vid.* stays for frames after Whisper, audio.* goes right after it.
            suffix = Path(upload["name"].split(" (")[0]).suffix.lower() or ".bin"
            src = path.rename(workdir / (("vid" if info["has_video"] else "audio") + suffix))
        else:
            src = sources[0]
            info = await asyncio.to_thread(media.probe_file, src, workdir)
        digest = await asyncio.to_thread(documents.sha256, src)
        return await asyncio.to_thread(
            pipeline.run_file, src, upload["name"], info, digest, progress, workdir=workdir,
            use_cache=job.use_cache, transcript_only=job.transcript_only, request_id=job.request_id,
            backend=job.backend, model=job.model,
            hide_cache_from=None if access.is_admin(job.user_id) else job.user_id)
    except memory.NeedsMemory:
        keep = True  # the file stays for the retry
        raise
    finally:
        if not keep:
            shutil.rmtree(workdir, ignore_errors=True)


async def _run_document(app: Application, job: state.Job, progress: sending.Progress) -> documents.DocResult:
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
                await _fetch_telegram_file(app, upload["file_id"], path, texts.TOO_BIG, documents.DocumentError)
            if job.cancel_reason:
                raise proc.ProcCancelled("cancelled")
        return await asyncio.to_thread(
            documents.run, upload, job.book_mode, job.chapter, path, workdir, progress, backend=job.backend,
            model=job.model, request_id=job.request_id,
            hide_cache_from=None if access.is_admin(job.user_id) else job.user_id, ocr_confirmed=job.ocr_ok)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)  # only the extracted text is kept


async def _run_voice(app: Application, job: state.Job, progress: sending.Progress) -> tts.VoiceResult:
    """Makes (or reuses) the voice message for a summary.

    A reused one is paced for a user who hasn't had it before (like other cached answers), off the worker.

    Raises:
        pipeline.PipelineError: The summary is no longer stored, or the speech failed.
    """
    spoken = db.get_spoken(job.voice_of)
    if not spoken:
        raise pipeline.PipelineError(texts.VOICE_TOO_OLD)
    text, lang, voice = spoken["text"], spoken["lang"], spoken["voice"]
    key = tts.key(text, lang, voice)
    db.update_request(job.request_id, status="processing")
    result = tts.VoiceResult(spoken["title"], text, lang, voice, key)
    estimate = tts.estimate(len(text))
    status = "🔊 Making the voice message…"
    if saved := db.get_voice(key, config.VOICE_CACHE_DAYS * limits.DAY):
        result.file_id, result.duration, result.cached = saved["file_id"], saved["duration"], True
        if not access.is_admin(job.user_id) and not db.user_saw_video(job.user_id, "voice", key, job.request_id):
            result.hold = [(status, min(estimate * pipeline.REPLAY_SHARE, pipeline.REPLAY_MAX))]
        return result
    progress(status, estimate)
    result.ogg = await make_voice(job, result, progress)
    return result


async def _run_ask(app: Application, job: state.Job, progress: sending.Progress) -> followup.AskResult:
    """Answers a follow-up question (in a thread: the AI call blocks).

    Raises:
        pipeline.PipelineError: The summary is too old, or the AI failed.
    """
    db.update_request(job.request_id, status="processing")
    progress(f"💬 {job.question[:60]}\n🧠 Thinking…", None)
    return await asyncio.to_thread(followup.answer, job.ask_of, job.question, job.user_id, job.backend, job.model)


async def _run_group(app: Application, job: state.Job, progress: sending.Progress) -> group.GroupResult:
    """Works on a several-link message (in a thread: downloads, Whisper and AI calls block)."""
    db.update_request(job.request_id, status="processing")
    admin = access.is_admin(job.user_id)
    return await asyncio.to_thread(
        group.run, job.urls, job.part_ids, progress, use_cache=job.use_cache, transcript_only=job.transcript_only,
        backend=job.backend, model=job.model, hide_cache_from=None if admin else job.user_id,
        again_limit_user=None if admin else job.user_id, numbers=job.numbers)


# Each job kind's runner: (app, job, progress) -> its result. Every JobKind must have one (tested).
RUNNERS = {state.JobKind.VIDEO: _run_video, state.JobKind.MEDIA: _run_media,
           state.JobKind.DOCUMENT: _run_document, state.JobKind.VOICE: _run_voice, state.JobKind.ASK: _run_ask,
           state.JobKind.GROUP: _run_group}


async def make_voice(job: state.Job, result: tts.VoiceResult,
                     status: Callable[[str, float | None], None] | None = None) -> Path:
    """Runs the speech in the sandbox (off the event loop, in its turn on the CPU); returns the .ogg in the job's
    work folder.

    Args:
        status: Callback (text, eta) for the job's status while it waits for its turn.
    """
    workdir = config.DATA_DIR / "work" / f"voice_{job.request_id}"
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)
    estimate = tts.estimate(len(result.text))

    def speak() -> tuple[Path, float]:
        """Speaks in this job's turn on the CPU; returns the file and the seconds the speech itself took."""
        with cpu.slot(status, "voice", estimate):
            started = time.monotonic()
            out = tts.synthesize(tts.split(result.text), workdir, voice=result.voice, lang=result.lang,
                                 speed=config.TTS_SPEED, timeout=estimate * 3 + 120)
            return out, time.monotonic() - started

    try:
        ogg, took = await asyncio.to_thread(speak)
    except proc.ProcCancelled:  # a ProcError, so a RuntimeError too: must stay a cancel
        shutil.rmtree(workdir, ignore_errors=True)
        raise
    except RuntimeError as e:
        shutil.rmtree(workdir, ignore_errors=True)
        raise pipeline.PipelineError("⚠️ Couldn't make the voice message. Please try again later.", detail=str(e))
    except BaseException:
        shutil.rmtree(workdir, ignore_errors=True)
        raise
    tts.record_speed(len(result.text), took)
    return ogg
