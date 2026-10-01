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
    """Everything the bot needs to reply about one video.

    Attributes:
        platform: "youtube" or "tiktok".
        video_id: The platform's id for the video.
        url: Canonical URL of the video.
        meta: Metadata from the yt-dlp probe (title, duration, is_carousel, ...).
        transcript: Transcript text with [m:ss] markers, or "" when there is none.
        transcript_source: "captions", "whisper-<model>", "tiktok-webvtt" or "none".
        language: Detected or caption language code, or "".
        summary: {title, is_clickbait, clickbait_answer, summary, _stats}, or None for /transcript runs.
        frames_used: Whether frames or slides were shown to the LLM.
        cached: Whether the result came from the database without new work.
        notes: Non-fatal problems hit along the way (for logs/debugging).
    """
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
    """A failure whose message is fit to show the user as-is."""
    pass


def _fmt_duration(sec: float) -> str:
    """Format seconds as m:ss, or h:mm:ss from one hour up.

    Args:
        sec: Duration in seconds; None or 0 gives "0:00".

    Returns:
        The formatted duration.
    """
    sec = int(sec or 0)
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}" if sec >= 3600 else f"{sec // 60}:{sec % 60:02d}"


# ---------- ETA estimates (seconds) ----------
# Rough models, deliberately simple: the measured per-backend speed in stats.json corrects them over time.

def _llm_load(chars: int, n_images: int) -> float:
    """Relative size of an LLM request, used to normalise measured LLM time.

    Args:
        chars: Transcript length in characters.
        n_images: Number of images attached.

    Returns:
        1.0 for an empty request, growing with transcript length and images.
    """
    # 40000 chars (~10k tokens) roughly doubles the time of a minimal request; each image adds ~10%.
    # Guesses that only need to be proportionate: stats.record() divides measured time by this factor.
    return 1 + chars / 40000 + n_images * 0.1


def _eta_llm(chars: int, n_images: int, backend: str | None = None) -> float:
    """Estimated seconds for one LLM turn.

    Args:
        chars: Transcript length in characters.
        n_images: Number of images attached.
        backend: LLM backend; defaults to config.LLM_BACKEND.

    Returns:
        Measured base time for the backend scaled by the request size.
    """
    # 25 s is the starting guess until the first real call on this machine has been measured.
    return stats.get(f"llm:{backend or config.LLM_BACKEND}", 25) * _llm_load(chars, n_images)


def _eta_audio(duration: float) -> float:
    """Estimated seconds to download a video's audio track.

    Args:
        duration: Video length in seconds.

    Returns:
        A few seconds of overhead plus a little per second of audio.
    """
    return 3 + duration / 200


def _eta_frames(duration: float) -> float:
    """Estimated seconds to download a video and grab frames from it.

    Args:
        duration: Video length in seconds.

    Returns:
        Fixed overhead plus time proportional to the video's length.
    """
    return 10 + duration / 20


def _chars_for(duration: float) -> int:
    """Expected transcript length for a video, for ETAs made before the transcript exists.

    Args:
        duration: Video length in seconds.

    Returns:
        Estimated number of transcript characters.
    """
    return int(duration * 15)  # ~15 transcript characters per second of speech


class Status:
    """Accumulates status lines under a header and reports them with an ETA."""

    def __init__(self, progress: Callable[..., None]):
        """Create an empty status.

        Args:
            progress: Callback taking (text, eta); the bot edits its status message with it.
        """
        self.progress, self.head, self.done = progress, "", []

    def show(self, current: str, eta: float | None = None) -> None:
        """Report the header, the finished steps and the current step.

        Args:
            current: The step in progress, e.g. "🧠 Summarizing…".
            eta: Estimated seconds until the summary is ready, or None if unknown.
        """
        self.progress("\n".join(filter(None, [self.head, *self.done, current])), eta)

    def ok(self, line: str) -> None:
        """Record a finished step; it appears from the next show() on.

        Args:
            line: The completed step, e.g. "✅ Transcript: YouTube captions (en)".
        """
        self.done.append(line)


def run(url: str, progress: Callable[..., None], *, use_cache: bool = True,
        transcript_only: bool = False, request_id: int | None = None, backend: str | None = None,
        model: str | None = None, again_limit_user: int | None = None) -> Result:
    """Summarize a YouTube or TikTok video, from the cache when possible.

    Summaries are cached per video and model, so each user gets the one their model wrote; a missing
    one is written from the saved transcript.

    Args:
        url: The link the user sent.
        progress: Callback taking (text, eta) for the live status message.
        use_cache: False for /again: ignore a cached summary and write a new one.
        transcript_only: For /transcript: stop after the transcript, no LLM call.
        request_id: The requests row to update with the resolved video and status.
        backend: The user's LLM backend; None = config.LLM_BACKEND.
        model: The user's model; None = that backend's default.
        again_limit_user: For /again by a non-admin: the user whose /again cooldown applies.

    Returns:
        The result to render.

    Raises:
        UnsupportedURL: The link isn't a YouTube or TikTok video.
        PipelineError: The video can't be processed (the message is shown to the user).
    """
    backend = backend or config.LLM_BACKEND
    model = model or summarize.default_model(backend)  # the cache key; a pinned model id when possible
    t0 = time.time()
    video = classify(url)

    if request_id:
        db.update_request(request_id, platform=video.platform, video_id=video.video_id, status="processing")
    if again_limit_user is not None and not use_cache:
        # Checked here, not when the link arrives: only now is the video's id known (short links are resolved
        # above). The requests table has every past /again, so the limit also survives restarts.
        last = db.last_again(again_limit_user, video.platform, video.video_id, request_id or 0)
        wait = (last or 0) + config.AGAIN_COOLDOWN_MIN * 60 - time.time()
        if last and wait > 0:
            ago = max(1, round((time.time() - last) / 60))
            raise PipelineError(f"⏳ You redid this video {ago} min ago; you can redo it again in "
                                f"{max(1, round(wait / 60))} min.")

    cached = db.get_video(video.platform, video.video_id)
    saved = db.get_summary(video.platform, video.video_id, backend, model) if model else None
    if cached and use_cache and cached["meta"] and (
            saved or (transcript_only and cached["transcript_source"])):
        return Result(video.platform, video.video_id, video.url, cached["meta"], cached["transcript"] or "",
                      cached["transcript_source"] or "none", cached["language"] or "",
                      saved["result"] if saved else None, bool(saved and saved["frames_used"]), cached=True)

    # The videos row exists from the start (status "processing") and is filled in as data arrives, so a
    # crash mid-way still leaves a record of what was attempted and how far it got.
    db.start_video(video.platform, video.video_id, video.url)
    try:
        return _process(video, progress, cached, transcript_only, t0, backend, model)
    except Exception as e:
        db.update_video(video.platform, video.video_id, status="failed", error=str(e)[:500])
        raise


def _process(video, progress, cached: dict | None, transcript_only: bool, t0: float,
             backend: str, model: str | None) -> Result:
    """Do the work for a video that isn't (fully) cached: lookup, transcript, frames, LLM.

    Args:
        video: The classified video (urls.Video).
        progress: Callback taking (text, eta) for the live status message.
        cached: The video's existing database row, whose transcript is reused, or None.
        transcript_only: Stop after the transcript.
        t0: Wall-clock start time, for the total shown in the footer.
        backend: LLM backend to use.
        model: Model to use; also the cache key the summary is saved under.

    Returns:
        The result to render.

    Raises:
        PipelineError: The video can't be loaded, is too long, has no slides, or the LLM failed.
    """
    st = Status(progress)
    timings: list[tuple[str, float]] = []  # (step, seconds), shown under the summary

    def took(label: str, since: float) -> None:
        """Record how long a step took, for the footer.

        Args:
            label: Step name shown to the user, e.g. "lookup" or "Whisper".
            since: time.monotonic() value when the step started.
        """
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
    shutil.rmtree(workdir, ignore_errors=True)  # leftovers from a crashed earlier run
    workdir.mkdir(parents=True)
    llm = summarize.llm_label(backend, model or "")  # replaced by the model that actually answers, below
    try:
        cues, source, lang, images, notes = [], "none", "", [], []
        # The probe detects carousels shared as /video/ links (no video formats), not just /photo/ URLs.
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
            # Reuse a cached transcript when re-running (/again) or writing another model's summary:
            # never fetch captions or run Whisper twice. It is already formatted, so it becomes one cue.
            cues = [(0.0, cached["transcript"])]
            source, lang = cached["transcript_source"], cached["language"]
            st.ok(f"✅ Transcript: from cache ({source})")
        else:
            t = time.monotonic()
            cues, source, lang = _transcript(video, meta, workdir, st, notes, transcript_only)
            took({"captions": "captions", "tiktok-webvtt": "Whisper + captions"}.get(source, "Whisper"), t)

        # A single cue is a cached, already formatted transcript; don't add markers to it again.
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

        thumb = media.download_thumbnail(meta, workdir)  # needed to judge clickbait thumbnails
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
            # Store the time per unit of request size, so future ETAs scale to each request.
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
                        # Same conversation: the transcript is already in context and in the provider's
                        # prompt cache, so only the frames are new.
                        summary = conv.add_frames(frames_)
                        took("update with frames", t)
                        images = frames_
                    except summarize.SummaryError as e:  # keep the transcript-only summary
                        log.warning("frames follow-up failed: %s", e)
                        notes.append(f"frames follow-up failed: {e}")
        except summarize.SummaryError as e:
            raise PipelineError(str(e))
        finally:
            conv.close()  # deletes the CLI session files; they're only needed for the follow-up turn
        summary["_stats"] = {"steps": timings, "total": time.time() - t0, "llm": llm,  # cached with the summary
                             "backend": backend, "model": model or conv.model}
        # Keyed by backend + model so users on different models don't overwrite each other's summaries.
        db.save_summary(video.platform, video.video_id, backend, model or conv.model, summary, bool(images))
        return _finish(video, meta, transcript, source, lang, summary, bool(images), notes, t0)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)  # keep no downloaded media


def _finish(video, meta, transcript, source, lang, summary, frames_used, notes, t0) -> Result:
    """Mark the video done and build the result.

    Args:
        video: The classified video (urls.Video).
        meta: Probe metadata.
        transcript: Formatted transcript text.
        source: Transcript source, e.g. "captions".
        lang: Transcript language.
        summary: The LLM's answer, or None for /transcript runs.
        frames_used: Whether frames or slides were shown to the LLM.
        notes: Non-fatal problems hit along the way.
        t0: Wall-clock start time, for the log line.

    Returns:
        The result to render.
    """
    db.update_video(video.platform, video.video_id, status="done", error=None)
    log.info("done %s/%s source=%s lang=%s frames=%s in %.0fs", video.platform, video.video_id, source,
             lang, frames_used, time.time() - t0)
    return Result(video.platform, video.video_id, video.url, meta, transcript, source, lang, summary,
                  frames_used, notes=notes)


def _speechless(meta: dict, transcript: str) -> bool:
    """Whether a video has no or hardly any speech, so its picture is the content.

    Args:
        meta: Probe metadata (for the duration).
        transcript: Formatted transcript; its [m:ss] markers aren't counted as words.

    Returns:
        True for fewer than 5 words, or fewer than 15 words per minute.
    """
    words = len(re.sub(r"\[\d+:\d\d\]", "", transcript).split())
    # Whisper often hears a few words in music or noise, hence a rate rather than "no words at all".
    # Normal speech is 100+ words per minute; 15 leaves room for slow talkers. The 0.25 min floor keeps
    # very short clips from passing on a single word.
    return words < 5 or words / max((meta.get("duration") or 0) / 60, 0.25) < 15


def _frames(video, meta, cues, moments: list[dict], workdir, st: Status, notes, why: str = "") -> list[tuple]:
    """Grab frames at the moments the LLM asked for, or sample the whole video.

    Args:
        video: The classified video (urls.Video).
        meta: Probe metadata.
        cues: Transcript cues [(start_seconds, text)]; phrase matches add extra moments.
        moments: The LLM's frame_moments [{t, why}]; empty = sample the whole video.
        workdir: The job's temp directory.
        st: Status to report progress to.
        notes: List that collects non-fatal problems.
        why: Reason shown in the status line; defaults to the LLM's reasons.

    Returns:
        [(jpeg_path, label)], or [] if the video couldn't be downloaded.
    """
    dur = meta["duration"] or 0
    # The LLM can return timestamps past the end or negative ones; drop them.
    times = sorted({float(m["t"]) for m in moments if 0 <= float(m.get("t", -1)) < max(dur, 1)})
    reasons = "; ".join(dict.fromkeys(str(m.get("why", "")) for m in moments if m.get("why")))[:120]
    # Dense sampling too when the LLM wants the whole video, or the video is short (cheap, catches
    # things shown briefly between the moments it named).
    sweep = not times or frames.is_short(meta)
    st.show(f"🎞 Grabbing frames: {why or reasons or 'LLM wants to see the video'}…",
            _eta_frames(dur) + _eta_llm(0, config.MAX_FRAMES))
    try:
        vid = media.download_video(video, workdir)
        # Phrase matches ("this book", "as you can see") back up the LLM's choice of moments.
        images = frames.extract(vid, dur, times + frames.regex_moments(cues), workdir, sweep=sweep)
        vid.unlink(missing_ok=True)
    except media.MediaError as e:
        notes.append(f"frames unavailable: {e}")
        st.ok("⚠️ Couldn't get video frames, keeping the transcript-only summary")
        return []
    st.ok(f"✅ {len(images)} frames grabbed")
    return images


def _whisper(video, meta, workdir, st: Status, notes, why: str, rest: float) -> tuple[list, str, str] | None:
    """Download a video's audio and transcribe it with Whisper.

    Args:
        video: The classified video (urls.Video).
        meta: Probe metadata.
        workdir: The job's temp directory.
        st: Status to report progress to.
        notes: List that collects non-fatal problems.
        why: Why Whisper is needed, shown in the status, e.g. "No captions".
        rest: Estimated seconds for the steps after transcription, added to the ETA.

    Returns:
        (cues, source, language), with cues empty when there is no speech; None when the audio
        couldn't be downloaded or has no audio stream.
    """
    dur = meta["duration"] or 0
    whisper = f"Whisper ({config.WHISPER_MODEL}, {config.WHISPER_DEVICE.upper()})"
    st.show(f"🎧 {why} → downloading audio for {whisper}…",
            _eta_audio(dur) + transcribe.estimate(dur) + rest)
    try:
        audio = media.download_audio(video, workdir)
    except media.MediaError as e:
        notes.append(f"audio download failed: {e}")
        return None
    # Some downloads (e.g. gallery-dl's h265 TikTok mp4s) have no audio track; Whisper would just fail.
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
    """Get a transcript from the cheapest working source.

    Args:
        video: The classified video (urls.Video).
        meta: Probe metadata.
        workdir: The job's temp directory.
        st: Status to report progress to.
        notes: List that collects non-fatal problems.
        transcript_only: No LLM step follows, so it's left out of the ETA.

    Returns:
        (cues, source, language); ([], "none", ...) when no transcript could be made.
    """
    dur = meta["duration"] or 0
    rest = 0 if transcript_only else _eta_llm(_chars_for(dur), 1)
    if video.platform == "youtube":
        # Captions take about a second; Whisper on this CPU takes ~0.3x the video's length.
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
