"""URL in, summary out. Blocking; the bot runs it in a worker thread.

`progress(text, eta)` is called at every stage: `text` is the full status to show, `eta` the estimated
seconds until the summary is ready (None when unknown).
"""
import logging
import re
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from . import config, db, frames, media, stats, summarize, transcribe
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


# ---------- ETA estimates (seconds) ----------

def _llm_load(chars: int, n_images: int) -> float:
    return 1 + chars / 40000 + n_images * 0.1


def _eta_llm(chars: int, n_images: int, backend: str | None = None) -> float:
    return stats.get(f"llm:{backend or config.LLM_BACKEND}", 25) * _llm_load(chars, n_images)


def _eta_audio(duration: float) -> float:
    return 3 + duration / 200


def _eta_frames(duration: float) -> float:
    return 10 + duration / 20


def _chars_for(duration: float) -> int:
    return int(duration * 15)  # ~15 transcript characters per second of speech


class Status:
    """Accumulates status lines under a header and reports them with an ETA."""

    def __init__(self, progress: Callable[..., None]):
        self.progress, self.head, self.done = progress, "", []

    def show(self, current: str, eta: float | None = None) -> None:
        self.progress("\n".join(filter(None, [self.head, *self.done, current])), eta)

    def ok(self, line: str) -> None:
        self.done.append(line)


def run(url: str, progress: Callable[..., None], *, use_cache: bool = True,
        transcript_only: bool = False, request_id: int | None = None, backend: str | None = None,
        model: str | None = None) -> Result:
    """backend/model: the user's LLM choice (None = defaults). Summaries are cached per video and model,
    so each user gets the one their model wrote; a missing one is written from the saved transcript.
    /again (use_cache=False) rewrites it."""
    backend = backend or config.LLM_BACKEND
    model = model or summarize.default_model(backend)  # the cache key; a pinned model id when possible
    t0 = time.time()
    video = classify(url)

    if request_id:
        db.update_request(request_id, platform=video.platform, video_id=video.video_id, status="processing")

    cached = db.get_video(video.platform, video.video_id)
    saved = db.get_summary(video.platform, video.video_id, backend, model) if model else None
    if cached and use_cache and cached["meta"] and (
            saved or (transcript_only and cached["transcript_source"])):
        return Result(video.platform, video.video_id, video.url, cached["meta"], cached["transcript"] or "",
                      cached["transcript_source"] or "none", cached["language"] or "",
                      saved["result"] if saved else None, bool(saved and saved["frames_used"]), cached=True)

    db.start_video(video.platform, video.video_id, video.url)
    try:
        return _process(video, progress, cached, transcript_only, t0, backend, model)
    except Exception as e:
        db.update_video(video.platform, video.video_id, status="failed", error=str(e)[:500])
        raise


def _process(video, progress, cached: dict | None, transcript_only: bool, t0: float,
             backend: str, model: str | None) -> Result:
    st = Status(progress)
    timings: list[tuple[str, float]] = []  # (step, seconds), shown under the summary

    def took(label: str, since: float) -> None:
        timings.append((label, time.monotonic() - since))

    st.show("🔎 Looking up the video…")
    t = time.monotonic()
    try:
        meta = media.probe(video)
        took("lookup", t)
        db.update_video(video.platform, video.video_id, meta=meta, title=meta["title"][:300])
    except media.MediaError as e:
        raise PipelineError(f"Couldn't load the video: {e}")
    dur = meta["duration"] or 0
    if video.kind == "photo" or meta.get("is_carousel"):
        st.head = f"🖼 {meta['title'][:80]} (photo post)"  # "duration" would be the music's
    else:
        st.head = f"🎬 {meta['title'][:80]} ({_fmt_duration(dur)})"
    if dur > config.MAX_DURATION_MIN * 60:
        raise PipelineError(f"Video is longer than {config.MAX_DURATION_MIN} min; skipping.")

    workdir = config.DATA_DIR / "work" / f"{video.platform}_{video.video_id}"
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)
    llm = summarize.llm_label(backend, model or "")  # replaced by the model that actually answers, below
    try:
        cues, source, lang, images, notes = [], "none", "", [], []
        is_carousel = video.kind == "photo" or bool(meta.get("is_carousel"))

        if is_carousel:
            # Photo post: the slides are the content; the audio is usually just a music track.
            st.show("🖼 Photo post: downloading the slides…", 10 + _eta_llm(0, 10, backend))
            t = time.monotonic()
            slides = media.download_carousel(video, workdir)
            if not slides:
                raise PipelineError("This photo post has no downloadable images (deleted, private, "
                                    "or region-locked).")
            images = [(media.to_jpeg(p, p.with_suffix(".conv.jpg")), f"slide {i}/{len(slides)}")
                      for i, p in enumerate(slides[:config.MAX_SLIDES], 1)]
            took(f"{len(images)} slides", t)
            st.ok(f"✅ Photo post: {len(images)} slides")
        elif cached and cached["transcript_source"] not in (None, "none"):
            # Reuse a cached transcript when re-running (/again): never fetch captions twice.
            cues = [(0.0, cached["transcript"])]
            source, lang = cached["transcript_source"], cached["language"]
            st.ok(f"✅ Transcript: from cache ({source})")
        else:
            t = time.monotonic()
            cues, source, lang = _transcript(video, meta, workdir, st, notes, transcript_only)
            took({"captions": "captions", "tiktok-webvtt": "Whisper + captions"}.get(source, "Whisper"), t)

        transcript = summarize.format_transcript(cues) if len(cues) > 1 else (cues[0][1] if cues else "")
        db.update_video(video.platform, video.video_id, transcript=transcript, transcript_source=source,
                        language=lang)
        if transcript_only:
            return _finish(video, meta, transcript, source, lang, None, False, notes, t0)

        # No (or hardly any) speech: the picture is the content, so look right away instead of
        # waiting for the LLM to ask. Saves the second LLM turn.
        if not is_carousel and _speechless(meta, transcript):
            t = time.monotonic()
            images = _frames(video, meta, cues, [], workdir, st, notes, why="no speech, looking at the video")
            took(f"{len(images)} frames", t)

        thumb = media.download_thumbnail(meta, workdir)
        first_images = ([(thumb, "thumbnail")] if thumb else []) + images  # slides or frames, if any
        conv = summarize.conversation(backend, model)
        try:
            st.show(f"🧠 Summarizing with {llm}…", _eta_llm(len(transcript), len(first_images), backend))
            t_llm = time.monotonic()
            answer = conv.start(meta, video.platform, transcript, source, lang, first_images)
            if conv.model:
                llm = summarize.llm_label(backend, conv.model)
                if not model:  # remember what the default resolves to, for labels and /models
                    stats.remember(f"model:{backend}", conv.model)
            took("summary", t_llm)
            stats.record(f"llm:{backend}",
                         (time.monotonic() - t_llm) / _llm_load(len(transcript), len(first_images)))
            summary = {k: answer[k] for k in summarize.SCHEMA["required"]}
            log.info("needs_frames=%s moments=%s", answer.get("needs_frames"), answer.get("frame_moments"))
            if answer.get("needs_frames") and not images:  # it already has slides/frames otherwise
                st.ok("✅ First summary written")
                t = time.monotonic()
                frames_ = _frames(video, meta, cues, answer.get("frame_moments") or [], workdir, st, notes)
                took(f"{len(frames_)} frames", t)
                if frames_:
                    st.show(f"🧠 {llm} is updating the summary with {len(frames_)} frames…",
                            _eta_llm(0, len(frames_), backend))
                    try:
                        t = time.monotonic()
                        summary = conv.add_frames(frames_)
                        took("update with frames", t)
                        images = frames_
                    except summarize.SummaryError as e:  # keep the transcript-only summary
                        log.warning("frames follow-up failed: %s", e)
                        notes.append(f"frames follow-up failed: {e}")
        except summarize.SummaryError as e:
            raise PipelineError(str(e))
        finally:
            conv.close()
        summary["_stats"] = {"steps": timings, "total": time.time() - t0, "llm": llm,  # cached with the summary
                             "backend": backend, "model": model or conv.model}
        db.save_summary(video.platform, video.video_id, backend, model or conv.model, summary, bool(images))
        return _finish(video, meta, transcript, source, lang, summary, bool(images), notes, t0)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)  # keep no downloaded media


def _finish(video, meta, transcript, source, lang, summary, frames_used, notes, t0) -> Result:
    db.update_video(video.platform, video.video_id, status="done", error=None)
    log.info("done %s/%s source=%s lang=%s frames=%s in %.0fs", video.platform, video.video_id, source,
             lang, frames_used, time.time() - t0)
    return Result(video.platform, video.video_id, video.url, meta, transcript, source, lang, summary,
                  frames_used, notes=notes)


def _speechless(meta: dict, transcript: str) -> bool:
    words = len(re.sub(r"\[\d+:\d\d\]", "", transcript).split())
    return words < 5 or words / max((meta.get("duration") or 0) / 60, 0.25) < 15


def _frames(video, meta, cues, moments: list[dict], workdir, st: Status, notes, why: str = "") -> list[tuple]:
    """Grab frames at the moments the LLM asked for (or sample the video). Returns [(path, label)]."""
    dur = meta["duration"] or 0
    times = sorted({float(m["t"]) for m in moments if 0 <= float(m.get("t", -1)) < max(dur, 1)})
    reasons = "; ".join(dict.fromkeys(str(m.get("why", "")) for m in moments if m.get("why")))[:120]
    # Dense sampling too when the LLM wants the whole video, or the video is short (cheap, catches
    # things shown briefly between the moments it named).
    sweep = not times or frames.is_short(meta)
    st.show(f"🎞 Grabbing frames: {why or reasons or 'LLM wants to see the video'}…",
            _eta_frames(dur) + _eta_llm(0, config.MAX_FRAMES))
    try:
        vid = media.download_video(video, workdir)
        images = frames.extract(vid, dur, times + frames.regex_moments(cues), workdir, sweep=sweep)
        vid.unlink(missing_ok=True)
    except media.MediaError as e:
        notes.append(f"frames unavailable: {e}")
        st.ok("⚠️ Couldn't get video frames, keeping the transcript-only summary")
        return []
    st.ok(f"✅ {len(images)} frames grabbed")
    return images


def _whisper(video, meta, workdir, st: Status, notes, why: str, rest: float) -> tuple[list, str, str] | None:
    dur = meta["duration"] or 0
    whisper = f"Whisper ({config.WHISPER_MODEL}, {config.WHISPER_DEVICE.upper()})"
    st.show(f"🎧 {why} → downloading audio for {whisper}…",
            _eta_audio(dur) + transcribe.estimate(dur) + rest)
    try:
        audio = media.download_audio(video, workdir)
    except media.MediaError as e:
        notes.append(f"audio download failed: {e}")
        return None
    if not media.has_audio_stream(audio):
        notes.append("downloaded file has no audio track")
        return None
    st.show(f"🗣 {why} → transcribing {_fmt_duration(dur)} of audio with {whisper}…",
            transcribe.estimate(dur) + rest)
    cues, lang, prob = transcribe.transcribe(str(audio))
    audio.unlink(missing_ok=True)
    log.info("whisper: %d segments, lang=%s p=%.2f", len(cues), lang, prob)
    st.ok(f"✅ Transcript: Whisper, language {lang}" if cues else "✅ Whisper: no speech found")
    return cues, f"whisper-{config.WHISPER_MODEL}", lang


def _transcript(video, meta, workdir, st: Status, notes, transcript_only: bool) -> tuple[list, str, str]:
    dur = meta["duration"] or 0
    rest = 0 if transcript_only else _eta_llm(_chars_for(dur), 1)
    if video.platform == "youtube":
        st.show("📝 Checking for YouTube captions…")
        if got := media.fetch_captions(video, meta, workdir):
            st.ok(f"✅ Transcript: YouTube captions ({got[1]})")
            return got[0], "captions", got[1]
        if got := _whisper(video, meta, workdir, st, notes, "No captions", rest):
            return got
        st.ok("⚠️ No transcript available")
        return [], "none", ""

    # TikTok: no transcript API. Audio + whisper first, then TikTok's own auto-captions.
    got = _whisper(video, meta, workdir, st, notes, "TikTok has no transcript", rest)
    if got and got[0]:
        return got
    if caps := media.fetch_captions(video, meta, workdir):
        st.ok(f"✅ Transcript: TikTok captions ({caps[1]})")
        return caps[0], "tiktok-webvtt", caps[1]
    return [], "none", got[2] if got else ""
