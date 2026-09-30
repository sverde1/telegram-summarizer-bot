"""URL in, summary out. Blocking; the bot runs it in a worker thread."""
import logging
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from . import cache, config, frames, media, summarize, transcribe
from .urls import classify

log = logging.getLogger(__name__)


@dataclass
class Result:
    platform: str
    video_id: str
    url: str
    meta: dict
    transcript: str
    transcript_source: str
    language: str
    summary: dict | None  # {title, is_clickbait, clickbait_answer, summary}
    frames_used: bool = False
    cached: bool = False
    notes: list[str] = field(default_factory=list)


class PipelineError(RuntimeError):
    pass


def _fmt_duration(sec: float) -> str:
    sec = int(sec or 0)
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}" if sec >= 3600 else f"{sec // 60}:{sec % 60:02d}"


def run(url: str, progress: Callable[[str], None], *, force_frames: bool | None = None,
        use_cache: bool = True, transcript_only: bool = False) -> Result:
    t0 = time.time()
    video = classify(url)

    cached = cache.get(video.platform, video.video_id)
    if cached and use_cache and (cached["result"] or transcript_only) and force_frames is None:
        return Result(video.platform, video.video_id, video.url, cached["meta"], cached["transcript"],
                      cached["transcript_source"], cached["language"], cached["result"],
                      cached["frames_used"], cached=True)

    progress("🔎 Looking up the video…")
    try:
        meta = media.probe(video)
    except media.MediaError as e:
        raise PipelineError(f"Couldn't load the video: {e}")
    head = f"🎬 {meta['title'][:80]} ({_fmt_duration(meta['duration'])})"
    if meta["duration"] > config.MAX_DURATION_MIN * 60:
        raise PipelineError(f"Video is longer than {config.MAX_DURATION_MIN} min; skipping.")

    workdir = config.DATA_DIR / "work" / f"{video.platform}_{video.video_id}"
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)
    try:
        cues, source, lang, images, notes = [], "none", "", [], []
        is_carousel = video.kind == "photo"

        # Reuse a cached transcript when re-running (/again, /frames): never fetch captions twice.
        if cached and cached["transcript_source"] != "none" and not is_carousel:
            cues = [(0.0, cached["transcript"])]
            source, lang = cached["transcript_source"], cached["language"]
        elif is_carousel:
            progress(f"{head}\n🖼 Downloading slides…")
            slides = media.download_carousel(video, workdir)
            if not slides:
                raise PipelineError("This photo post has no downloadable images (deleted, private, "
                                    "or region-locked).")
            images = [(media.to_jpeg(p, p.with_suffix(".conv.jpg")), f"slide {i}/{len(slides)}")
                      for i, p in enumerate(slides[:config.MAX_FRAMES], 1)]
        else:
            cues, source, lang = _transcript(video, meta, workdir, progress, head, notes)

        transcript = summarize.format_transcript(cues) if len(cues) > 1 else (cues[0][1] if cues else "")
        if transcript_only:
            return _finish(video, meta, transcript, source, lang, None, False, notes, t0)

        if not is_carousel:
            use, reasons = frames.decide(meta, cues, force_frames)
            log.info("frames=%s reasons=%s", use, reasons)
            if use:
                progress(f"{head}\n✓ transcript ({source})\n🎞 Pulling frames: {'; '.join(reasons)}…")
                try:
                    vid = media.download_video(video, workdir)
                    images = frames.extract(vid, cues, meta["duration"], workdir)
                    vid.unlink(missing_ok=True)
                except media.MediaError as e:
                    notes.append(f"frames unavailable: {e}")

        progress(f"{head}\n✓ transcript ({source})"
                 + (f"\n✓ {len(images)} images" if images else "") + "\n🧠 Summarizing…")
        thumb = media.download_thumbnail(meta, workdir)
        try:
            summary = summarize.summarize(meta, video.platform, transcript, source, lang, thumb, images)
        except summarize.SummaryError as e:
            raise PipelineError(str(e))
        return _finish(video, meta, transcript, source, lang, summary, bool(images), notes, t0)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)  # keep no downloaded media


def _finish(video, meta, transcript, source, lang, summary, frames_used, notes, t0) -> Result:
    old = cache.get(video.platform, video.video_id)
    cache.put(video.platform, video.video_id, meta=meta, transcript=transcript, transcript_source=source,
              language=lang, result=summary or (old or {}).get("result"), frames_used=frames_used)
    log.info("done %s/%s source=%s lang=%s frames=%s in %.0fs", video.platform, video.video_id, source,
             lang, frames_used, time.time() - t0)
    return Result(video.platform, video.video_id, video.url, meta, transcript, source, lang, summary,
                  frames_used, notes=notes)


def _whisper(video, workdir, progress, head, notes) -> tuple[list, str, str] | None:
    progress(f"{head}\n🎧 Downloading audio…")
    try:
        audio = media.download_audio(video, workdir)
    except media.MediaError as e:
        notes.append(f"audio download failed: {e}")
        return None
    if not media.has_audio_stream(audio):
        notes.append("downloaded file has no audio track")
        return None
    progress(f"{head}\n🗣 Transcribing audio ({config.WHISPER_MODEL} on {config.WHISPER_DEVICE})…")
    cues, lang, prob = transcribe.transcribe(str(audio))
    audio.unlink(missing_ok=True)
    log.info("whisper: %d segments, lang=%s p=%.2f", len(cues), lang, prob)
    return cues, f"whisper-{config.WHISPER_MODEL}", lang


def _transcript(video, meta, workdir, progress, head, notes) -> tuple[list, str, str]:
    if video.platform == "youtube":
        progress(f"{head}\n📝 Fetching captions…")
        if got := media.fetch_captions(video, meta, workdir):
            return got[0], "captions", got[1]
        if got := _whisper(video, workdir, progress, head, notes):
            return got
        return [], "none", ""

    # TikTok: no transcript API. Audio + whisper first, then TikTok's own auto-captions.
    got = _whisper(video, workdir, progress, head, notes)
    if got and got[0]:
        return got
    if caps := media.fetch_captions(video, meta, workdir):
        return caps[0], "tiktok-webvtt", caps[1]
    return [], "none", got[2] if got else ""
