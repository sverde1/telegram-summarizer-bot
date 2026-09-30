"""URL in, summary out. Blocking; the bot runs it in a worker thread.

`progress(text, eta)` is called at every stage: `text` is the full status to show, `eta` the estimated
seconds until the summary is ready (None when unknown).
"""
import logging
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from . import cache, config, frames, media, stats, summarize, transcribe
from .urls import classify

log = logging.getLogger(__name__)

BACKEND_NAMES = {"codex": "Codex (ChatGPT)", "claude-code": "Claude Code", "api": "Claude API"}


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


def _eta_llm(chars: int, n_images: int) -> float:
    return stats.get(f"llm:{config.LLM_BACKEND}", 25) * _llm_load(chars, n_images)


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
        transcript_only: bool = False) -> Result:
    t0 = time.time()
    video = classify(url)

    cached = cache.get(video.platform, video.video_id)
    if cached and use_cache and (cached["result"] or transcript_only):
        return Result(video.platform, video.video_id, video.url, cached["meta"], cached["transcript"],
                      cached["transcript_source"], cached["language"], cached["result"],
                      cached["frames_used"], cached=True)

    st = Status(progress)
    st.show("🔎 Looking up the video…")
    try:
        meta = media.probe(video)
    except media.MediaError as e:
        raise PipelineError(f"Couldn't load the video: {e}")
    dur = meta["duration"] or 0
    st.head = f"🎬 {meta['title'][:80]} ({_fmt_duration(dur)})"
    if dur > config.MAX_DURATION_MIN * 60:
        raise PipelineError(f"Video is longer than {config.MAX_DURATION_MIN} min; skipping.")

    workdir = config.DATA_DIR / "work" / f"{video.platform}_{video.video_id}"
    shutil.rmtree(workdir, ignore_errors=True)
    workdir.mkdir(parents=True)
    llm = BACKEND_NAMES.get(config.LLM_BACKEND, config.LLM_BACKEND)
    try:
        cues, source, lang, images, notes = [], "none", "", [], []
        is_carousel = video.kind == "photo"

        # Reuse a cached transcript when re-running (/again, /frames): never fetch captions twice.
        if cached and cached["transcript_source"] != "none" and not is_carousel:
            cues = [(0.0, cached["transcript"])]
            source, lang = cached["transcript_source"], cached["language"]
            st.ok(f"✅ Transcript: from cache ({source})")
        elif is_carousel:
            st.show("🖼 Photo post: downloading slides…", 10 + _eta_llm(0, 10))
            slides = media.download_carousel(video, workdir)
            if not slides:
                raise PipelineError("This photo post has no downloadable images (deleted, private, "
                                    "or region-locked).")
            images = [(media.to_jpeg(p, p.with_suffix(".conv.jpg")), f"slide {i}/{len(slides)}")
                      for i, p in enumerate(slides[:config.MAX_FRAMES], 1)]
            st.ok(f"✅ {len(images)} slides downloaded")
        else:
            cues, source, lang = _transcript(video, meta, workdir, st, notes, transcript_only)

        transcript = summarize.format_transcript(cues) if len(cues) > 1 else (cues[0][1] if cues else "")
        if transcript_only:
            return _finish(video, meta, transcript, source, lang, None, False, notes, t0)

        if not is_carousel:
            images = _frames(video, meta, cues, transcript, workdir, st, notes, llm)

        n_img = len(images) + 1  # + thumbnail
        st.show(f"🧠 Summarizing with {llm}…", _eta_llm(len(transcript), n_img))
        thumb = media.download_thumbnail(meta, workdir)
        t_llm = time.monotonic()
        try:
            summary = summarize.summarize(meta, video.platform, transcript, source, lang, thumb, images)
        except summarize.SummaryError as e:
            raise PipelineError(str(e))
        stats.record(f"llm:{config.LLM_BACKEND}",
                     (time.monotonic() - t_llm) / _llm_load(len(transcript), n_img))
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


def _frames(video, meta, cues, transcript, workdir, st: Status, notes, llm) -> list[tuple]:
    """Decide which moments to look at, then grab frames there. Returns [(path, label)]."""
    dur = meta["duration"] or 0
    short = frames.is_short(meta)
    no_speech = frames.speech_wpm(meta, cues) < 15
    moments = frames.regex_moments(cues)
    if short or no_speech:
        why = "short video" if short else "little or no speech"
    else:
        st.show(f"🔍 Asking {llm} which moments show something on screen…",
                _eta_llm(len(transcript), 0) * 2 + _eta_frames(dur))
        try:
            tri = summarize.triage(meta, cues)
            log.info("triage: %s", tri)
            moments += [float(m["t"]) for m in tri["moments"] if 0 <= float(m["t"]) < dur]
        except (summarize.SummaryError, KeyError, TypeError, ValueError) as e:
            log.warning("triage failed, using phrase matching only: %s", e)
        if not moments:
            st.ok("✅ Nothing important shown on screen, skipping frames")
            return []
        why = f"{len(moments)} on-screen moments"
    st.show(f"🎞 Looking at the video ({why})…",
            _eta_frames(dur) + _eta_llm(len(transcript), config.MAX_FRAMES))
    try:
        vid = media.download_video(video, workdir)
        images = frames.extract(vid, dur, moments, workdir, sweep=short or no_speech)
        vid.unlink(missing_ok=True)
    except media.MediaError as e:
        notes.append(f"frames unavailable: {e}")
        st.ok("⚠️ Couldn't get video frames, continuing without")
        return []
    st.ok(f"✅ {len(images)} frames selected ({why})")
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
