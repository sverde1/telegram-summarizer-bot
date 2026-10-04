"""URL in, summary out. Blocking; the bot runs it in a worker thread.

`progress(text, eta)` is called at every stage: `text` is the full status to show, `eta` the estimated
seconds until the summary is ready (None when unknown).
"""
import contextlib
import logging
import re
import shutil
import tempfile
import threading
from pathlib import Path
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field

from . import config, cpu, db, frames, media, memory, proc, stats, summarize, transcribe
from .results import JobResult
from .urls import PLATFORM_NAMES, Video, classify

log = logging.getLogger(__name__)

@dataclass
class Result(JobResult):
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
        notes: Non-fatal problems hit along the way (for logs/debugging).

    The caching and pacing fields come from JobResult. On a fresh result, replay_steps include the paced
    transcript step; the footer shows them, so what the user waited and what the footer says always agree.
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
    notes: list[str] = field(default_factory=list)

    def head(self) -> str:
        """The status line naming the video, photo post or sent recording."""
        return status_head(self.platform, self.url, self.meta)

    def work_seconds(self) -> float | None:
        """The worker's time for the summary; None for transcript-only results (they'd skew the estimate)."""
        return (self.summary or {}).get("_stats", {}).get("total")

    def llm_label(self) -> str:
        """The model that answered, as stored with the summary."""
        return (self.summary or {}).get("_stats", {}).get("llm", "")


class PipelineError(RuntimeError):
    """The video can't be summarized. str() is a message fit for the user; `detail` has the technical cause
    (for the admin and the log), never shown to other users.
    """

    def __init__(self, message: str, detail: str | None = None):
        """Stores the user message and the optional technical detail."""
        super().__init__(message)
        self.detail = detail


class Blocked(PipelineError):
    """The platform is blocking this server's downloads; the admins are told (see bot)."""

    def __init__(self, message: str, detail: str | None = None, platform: str = ""):
        """Stores the user message, the raw error and the platform that blocks."""
        super().__init__(message, detail)
        self.platform = platform


def fmt_duration(sec: float) -> str:
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

def llm_load(chars: int, n_images: int) -> float:
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


def eta_llm(chars: int, n_images: int, backend: str | None = None) -> float:
    """Estimated seconds for one LLM turn.

    Args:
        chars: Transcript length in characters.
        n_images: Number of images attached.
        backend: LLM backend; defaults to config.LLM_BACKEND.

    Returns:
        Measured base time for the backend scaled by the request size.
    """
    # 25 s is the starting guess until the first real call on this machine has been measured.
    return stats.get(f"llm:{backend or config.LLM_BACKEND}", 25) * llm_load(chars, n_images)


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


# A first-time requester of reused work (a cached answer, or a saved transcript) waits this share of what a
# fresh run takes, at most REPLAY_MAX seconds: an instant answer would give away that someone else sent the
# video before, and half a run still looks like a normal, quick one.
REPLAY_SHARE = 0.5
REPLAY_MAX = 120
CAPTIONS_SECONDS = 3.0  # rough time to fetch captions (not measured: it's a single quick request)
LOOKUP_SECONDS = 3.0    # rough time for the lookup, for summaries saved before timings were recorded
TRANSCRIPT_STEPS = {"captions", "Whisper", "Whisper + captions"}  # took() labels of the transcript step


def _transcript_step(source: str, duration: float) -> tuple[str, float]:
    """The transcript step a fresh run of this video would show, with its estimated time.

    Args:
        source: The transcript source (captions, whisper-<model>, tiktok-webvtt, none).
        duration: Video length in seconds.

    Returns:
        (took() label, seconds).
    """
    label = {"captions": "captions", "tiktok-webvtt": "Whisper + captions"}.get(source, "Whisper")
    if label == "captions":
        return label, CAPTIONS_SECONDS
    return label, _eta_audio(duration) + transcribe.estimate(duration)


def replay_stage(name: str, llm: str) -> str:
    """The status line a real run shows for a step (see the took() labels in _process)."""
    if name == "lookup":
        return "🔎 Looking up the video…"
    if name == "reading the file":
        return "📄 Reading the file…"
    if name == "captions":
        return "📝 Checking for YouTube captions…"
    if name.startswith("Whisper"):
        return "🗣 Transcribing the audio with Whisper…"
    if name.endswith("slides"):
        return "🖼 Photo post: downloading the slides…"
    if name.endswith("frames"):
        return "🎞 Grabbing frames…"
    if name == "update with frames":
        return f"🧠 {llm} is updating the summary with frames…"
    return f"🧠 Summarizing with {llm}…"


def _full_steps(r: Result, transcript_only: bool, backend: str) -> list[tuple[str, float]]:
    """Every step a fresh run producing this result would show, with its time.

    Recorded times are used where the original run has them; missing steps are estimated: summaries saved
    before timings were recorded have none, a summary written from a reused transcript has no transcript
    step, and a /transcript has no summary of its own.
    """
    stats_ = (r.summary or {}).get("_stats") or {}
    steps = [(name, float(sec)) for name, sec in stats_.get("steps", [])]
    if transcript_only:  # what a /transcript run does: lookup, transcript (or slides), nothing after
        steps = [(n, s) for n, s in steps if n == "lookup" or n in TRANSCRIPT_STEPS or n.endswith("slides")]
    if not any(n == "lookup" for n, _ in steps):
        steps.insert(0, ("lookup", LOOKUP_SECONDS))
    if not r.meta.get("is_carousel") and not any(n in TRANSCRIPT_STEPS for n, _ in steps):
        steps.insert(1, _transcript_step(r.transcript_source, r.meta.get("duration") or 0))
    if not transcript_only and not any(n == "summary" for n, _ in steps):
        steps.append(("summary", eta_llm(len(r.transcript), 1, backend)))
    return steps


def plan_replay(r: Result, transcript_only: bool, backend: str) -> None:
    """Sets a cached result's replay: all of a fresh run's steps, scaled to REPLAY_SHARE (at most REPLAY_MAX).

    Computed once per job, so the bot's playback and the footer use the same numbers.
    """
    steps = _full_steps(r, transcript_only, backend)
    total = sum(sec for _, sec in steps)
    delay = min(total * REPLAY_SHARE, REPLAY_MAX)
    scale = delay / total if total else 0
    r.replay_steps = [(name, sec * scale) for name, sec in steps]
    r.replay_total = delay


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
        self.frozen = False  # keep the stage on screen (pacing): later shows only check for a cancel

    def show(self, current: str, eta: float | None = None) -> None:
        """Report the header, the finished steps and the current step.

        Args:
            current: The step in progress, e.g. "🧠 Summarizing…".
            eta: Estimated seconds until the summary is ready, or None if unknown.
        """
        # Every stage change is a cancellation checkpoint: a removed user's job stops before its next step.
        proc.check_cancelled()
        if not self.frozen:
            self.progress(self.text(current), eta)

    def text(self, current: str) -> str:
        """The full status text with `current` as the stage in progress."""
        return "\n".join(filter(None, [self.head, *self.done, current]))

    def ok(self, line: str) -> None:
        """Record a finished step; it appears from the next show() on.

        Args:
            line: The completed step, e.g. "✅ Transcript: YouTube captions (en)".
        """
        self.done.append(line)


_busy: set[tuple[str, str]] = set()  # videos and documents being worked on right now (see one_at_a_time)
_busy_changed = threading.Condition()


@contextmanager
def one_at_a_time(key: tuple[str, str], on_wait: Callable[[], None]):
    """Lets one job at a time work on a video or document; others wait (cancellably) until it's done.

    Parallel jobs for the same thing would download, transcribe or summarize it twice and interleave their
    writes to its database row; after the wait, the second job finds the first one's work in the cache.

    Args:
        key: (platform, video id), or ("document", sha256).
        on_wait: Called once if this job has to wait (to show it in the status).

    Raises:
        proc.ProcCancelled: The job was cancelled while waiting.
    """
    with _busy_changed:
        if key in _busy:
            on_wait()
        while key in _busy:
            _busy_changed.wait(1)
            proc.check_cancelled()
        _busy.add(key)
    try:
        yield
    finally:
        with _busy_changed:
            _busy.discard(key)
            _busy_changed.notify_all()


def run(url: str, progress: Callable[..., None], *, use_cache: bool = True,
        transcript_only: bool = False, request_id: int | None = None, backend: str | None = None,
        model: str | None = None, again_limit_user: int | None = None,
        hide_cache_from: int | None = None) -> Result:
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
        hide_cache_from: A non-admin requester who must not learn that someone else processed this video
            before (if they haven't requested it themselves): status lines then don't mention the cache.

    Returns:
        The result to render.

    Raises:
        UnsupportedURL: The link isn't a YouTube or TikTok video.
        PipelineError: The video can't be processed (the message is shown to the user).
    """
    return _run(classify(url), progress, use_cache=use_cache, transcript_only=transcript_only,
                request_id=request_id, backend=backend, model=model, again_limit_user=again_limit_user,
                hide_cache_from=hide_cache_from)


def status_head(platform: str, url: str, meta: dict, photo: bool = False) -> str:
    """The status line naming what a job works on.

    Args:
        platform: The video's platform; "file" for a recording a user sent (its url is its label).
        url: The video's URL, or a sent file's label (e.g. "🎤 Voice message").
        meta: Its metadata (title, duration, is_carousel).
        photo: Whether it is a photo post even if the metadata doesn't say so (a /photo/ link).

    Returns:
        E.g. "🎬 Title (12:34)", "🖼 Title (photo post)" or "🎤 Voice message (0:42)".
    """
    if platform == "file":
        return f"{url[:80]} ({fmt_duration(meta.get('duration') or 0)})"
    if photo or meta.get("is_carousel"):
        return f"🖼 {meta.get('title', '')[:80]} (photo post)"  # "duration" would be the music's
    return f"🎬 {meta.get('title', '')[:80]} ({fmt_duration(meta.get('duration') or 0)})"


def run_file(src: Path, label: str, info: dict, sha256: str, progress: Callable[..., None], *, workdir: Path,
             use_cache: bool = True, transcript_only: bool = False, request_id: int | None = None,
             backend: str | None = None, model: str | None = None,
             hide_cache_from: int | None = None) -> Result:
    """Summarizes a voice message, audio or video file a user sent, like a video from a link.

    The file is identified by its SHA-256 (a repeat reuses its transcript and summaries). Its own name never
    goes into the shared cache: a later sender of the same file must not see the first sender's file name, so
    the stored metadata has a neutral title and the name lives only in this request's label.

    Args:
        src: The file, already in `workdir` (named vid.* when it has video, audio.* otherwise).
        label: What this request calls it, e.g. "🎤 Voice message" or "🎬 holiday.mp4".
        info: media.probe_file's result (duration, has_audio, has_video).
        sha256: The file's hash.
        progress: Callback (text, eta) for the status message.
        workdir: The job's directory, owned by the caller (kept when the job has to wait for memory).
        use_cache / transcript_only / request_id / backend / model / hide_cache_from: as for run().

    Raises:
        PipelineError: The file can't be summarized (message for the user).
        memory.NeedsMemory: Not enough free memory to transcribe it right now.
    """
    kind = "video" if info["has_video"] else "audio"
    video = Video("file", sha256, label, kind)
    neutral = "Video file" if kind == "video" else "Audio file"
    meta = {"id": sha256, "title": neutral, "duration": info["duration"], "uploader": "", "upload_date": "",
            "description": "", "thumbnail": "", "language": "", "subtitles": {}, "auto_captions": [],
            "music": "", "is_carousel": False, "has_audio": info["has_audio"], "has_video": info["has_video"]}
    return _run(video, progress, use_cache=use_cache, transcript_only=transcript_only, request_id=request_id,
                backend=backend, model=model, again_limit_user=None, hide_cache_from=hide_cache_from,
                workdir=workdir, file_meta=meta)


def _run(video, progress: Callable[..., None], *, use_cache: bool, transcript_only: bool,
         request_id: int | None, backend: str | None, model: str | None, again_limit_user: int | None,
         hide_cache_from: int | None, workdir: Path | None = None, file_meta: dict | None = None) -> Result:
    """The shared body of run() and run_file(): cache lookup, pacing, processing, the videos row's status."""
    backend = backend or config.LLM_BACKEND
    model = model or summarize.default_model(backend)  # the cache key; a pinned model id when possible
    t0 = time.time()
    cpu.track()  # waits for the CPU slot are left out of this job's timings

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

    # One job per video at a time: a second request for it waits, then finds the first one's work cached
    # (no second Whisper or summary, no interleaved writes to the videos row).
    with one_at_a_time((video.platform, video.video_id), lambda: progress("⏳ Processing…", None)):
        hide_cache = bool(hide_cache_from) and not db.user_saw_video(hide_cache_from, video.platform,
                                                                     video.video_id, request_id or 0)
        cached = db.get_video(video.platform, video.video_id)
        saved = db.get_summary(video.platform, video.video_id, backend, model) if model else None
        if cached and use_cache and cached["meta"] and (
                saved or (transcript_only and cached["transcript_source"])):
            result = Result(video.platform, video.video_id, video.url, cached["meta"], cached["transcript"] or "",
                            cached["transcript_source"] or "none", cached["language"] or "",
                            saved["result"] if saved else None, bool(saved and saved["frames_used"]), cached=True)
            if hide_cache:
                plan_replay(result, transcript_only, backend)
            return result

        # The videos row exists from the start (status "processing") and is filled in as data arrives, so a
        # crash mid-way still leaves a record of what was attempted and how far it got.
        # A file's name stays out of the shared row (see run_file).
        db.start_video(video.platform, video.video_id, "" if file_meta else video.url)
        try:
            return _process(video, progress, cached, transcript_only, t0, backend, model, hide_cache,
                            workdir=workdir, file_meta=file_meta)
        except proc.ProcCancelled:
            # The video itself is fine; only this request was stopped. Don't record it as a failed video.
            db.update_video(video.platform, video.video_id, status="cancelled", error=None)
            raise
        except memory.NeedsMemory:
            db.update_video(video.platform, video.video_id, status="waiting", error=None)  # will be retried
            raise
        except media.Blocked as e:
            db.update_video(video.platform, video.video_id, status="failed", error=str(e)[:500])
            name = PLATFORM_NAMES.get(video.platform, video.platform)
            raise Blocked(f"🚫 {name} is currently blocking downloads from this bot's server (too many requests). "
                          "Please try again later.", detail=str(e), platform=video.platform)
        except Exception as e:
            db.update_video(video.platform, video.video_id, status="failed", error=str(e)[:500])
            raise


def _process(video, progress, cached: dict | None, transcript_only: bool, t0: float,
             backend: str, model: str | None, hide_cache: bool = False, workdir: Path | None = None,
             file_meta: dict | None = None) -> Result:
    """Do the work for a video that isn't (fully) cached: lookup, transcript, frames, LLM.

    Args:
        video: The classified video (urls.Video).
        progress: Callback taking (text, eta) for the live status message.
        cached: The video's existing database row, whose transcript is reused, or None.
        transcript_only: Stop after the transcript.
        t0: Wall-clock start time, for the total shown in the footer.
        backend: LLM backend to use.
        model: Model to use; also the cache key the summary is saved under.
        hide_cache: Don't mention the cache in status lines (see run()).
        workdir: A directory the caller owns (a file a user sent is already in it); it is neither wiped nor
            deleted here. None: a fresh one for this video, deleted at the end.
        file_meta: For a file a user sent: its metadata (no lookup; every ffmpeg run in the sandbox).

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
        timings.append((label, time.monotonic() - since - cpu.waited(since)))  # waiting for a turn isn't work

    is_file = file_meta is not None
    owns_workdir = workdir is None
    if owns_workdir:
        # A new directory per job: two jobs for the same video must not share (and delete) each other's files.
        (config.DATA_DIR / "work").mkdir(parents=True, exist_ok=True)
        workdir = Path(tempfile.mkdtemp(dir=config.DATA_DIR / "work", prefix=f"{video.platform}_{video.video_id}_"))
    # A saved transcript makes the lookup unnecessary: its metadata is stored with it, and nothing needs
    # downloading unless the LLM asks for frames (then the video download looks it up itself).
    reuse_meta = bool(not is_file and cached and cached.get("meta")
                      and cached["transcript_source"] not in (None, "none"))
    owed_lookup = 0.0  # a first-time requester is still shown a lookup step (paced, see Result.hold)
    try:
        if is_file:  # nothing to look up: the caller read the file (sandboxed)
            meta = file_meta
            db.update_video(video.platform, video.video_id, meta=meta, title=meta["title"])
        elif reuse_meta:
            meta = cached["meta"]
            if hide_cache:
                st.show("🔎 Looking up the video…")
                owed_lookup = LOOKUP_SECONDS * REPLAY_SHARE
        else:
            st.show("🔎 Looking up the video…")
            t = time.monotonic()
            try:
                # The full answer is kept for this job's later yt-dlp calls (captions, downloads).
                meta = media.probe(video, workdir / media.INFO_JSON)
                took("lookup", t)
                db.update_video(video.platform, video.video_id, meta=meta, title=meta["title"][:300])
            except media.Blocked:
                raise  # handled in run(): a ban has its own message and admin notice
            except media.MediaError as e:
                raise PipelineError(media.describe(e), detail=str(e))
        dur = meta["duration"] or 0
        what = "This recording" if is_file else "Video"
        st.head = status_head(video.platform, video.url, meta, photo=video.kind == "photo")
        if dur > config.MAX_DURATION_MIN * 60:
            raise PipelineError(f"{what} is longer than {config.MAX_DURATION_MIN} min; skipping.")
        if not dur and not (video.kind == "photo" or meta.get("is_carousel")):
            # Without a length there's no limit on the download (and the length check above can't work).
            # Carousels are fine: their "duration" is just the background music's, if any.
            raise PipelineError(f"Couldn't determine this {'recording' if is_file else 'video'}'s length, so it "
                                "can't be processed.")
    except BaseException:
        if owns_workdir:
            shutil.rmtree(workdir, ignore_errors=True)
        raise
    llm = summarize.llm_label(backend, model or "")  # replaced by the model that actually answers, below
    # The thumbnail (for the clickbait check) downloads while the transcript is made; not for /transcript.
    side = proc.pool(1)
    thumb_job = None if transcript_only else side.submit(media.download_thumbnail, meta, workdir)
    paced: tuple[str, float] | None = None  # the paced transcript step shown to a first-time requester
    try:
        cues, source, lang, images, notes = [], "none", "", [], []
        # The probe detects carousels shared as /video/ links (no video formats), not just /photo/ URLs.
        is_carousel = video.kind == "photo" or bool(meta.get("is_carousel"))

        if is_carousel:
            # Photo post: the slides are the content; the audio is usually just a music track.
            st.show("🖼 Photo post: downloading the slides…", 10 + eta_llm(0, 10, backend))
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
            if hide_cache:
                # An instant transcript would tell a first-time requester that someone else sent this video
                # before: show the step a fresh run takes, at the pace of a cached answer. The stage stays on
                # screen while the real work runs (frozen); the time still owed afterwards is waited out by the
                # bot off the worker (Result.hold), so nobody else's job waits for this pause.
                label, eta = _transcript_step(source, dur)
                pause = min(eta * REPLAY_SHARE, REPLAY_MAX)
                st.show(replay_stage(label, llm), pause + eta_llm(len(cached["transcript"]), 1, backend))
                st.frozen = True
                paced = (label, pause)
                st.ok("✅ Transcript ready")
            else:
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
        if is_file and not meta.get("has_video") and not transcript.strip():
            raise PipelineError("🔇 No speech found in this recording.")

        # No (or hardly any) speech: the picture is the content, so look right away instead of
        # waiting for the LLM to ask. Saves the second LLM turn.
        if not is_carousel and meta.get("has_video", True) and _speechless(meta, transcript):
            t = time.monotonic()
            images = _frames(video, meta, cues, [], workdir, st, notes, why="no speech, looking at the video")
            took(f"{len(images)} frames", t)

        try:
            thumb = thumb_job.result(timeout=35) if thumb_job else None  # needed to judge clickbait thumbnails
        except Exception:  # noqa: BLE001  (the thumbnail is optional)
            thumb = None
        if thumb is None and reuse_meta and meta.get("thumbnail"):
            # Stored metadata may hold an expired thumbnail URL (TikTok signs them): look the video up again.
            try:
                thumb = media.download_thumbnail(media.probe(video, workdir / media.INFO_JSON), workdir)
            except media.Blocked:
                raise
            except media.MediaError as e:
                notes.append(f"thumbnail lookup failed: {e}")
        first_images = ([(thumb, "thumbnail")] if thumb else []) + images  # slides or frames, if any
        conv = summarize.conversation(backend, model)
        try:
            st.show(f"🧠 Summarizing with {llm}…", eta_llm(len(transcript), len(first_images), backend))
            t_llm = time.monotonic()
            answer = conv.start(meta, video.platform, transcript, source, lang, first_images)
            proc.check_cancelled()  # an API call can't be interrupted; at least don't go on after it
            if conv.model:
                llm = summarize.llm_label(backend, conv.model)
                if not model:  # remember what the default resolves to, for labels and /models
                    stats.remember(f"model:{backend}", conv.model)
            took("summary", t_llm)
            # Store the time per unit of request size, so future ETAs scale to each request.
            stats.record(f"llm:{backend}",
                         (time.monotonic() - t_llm) / llm_load(len(transcript), len(first_images)))
            summary = {k: answer[k] for k in summarize.SCHEMA["required"]}
            log.info("needs_frames=%s moments=%s", answer.get("needs_frames"), answer.get("frame_moments"))
            # Not when it already has slides or frames.
            if answer.get("needs_frames") and not images and meta.get("has_video", True):
                st.ok("✅ First summary written")
                t = time.monotonic()
                frames_ = _frames(video, meta, cues, answer.get("frame_moments") or [], workdir, st, notes)
                took(f"{len(frames_)} frames", t)
                if frames_:
                    st.show(f"🧠 {llm} is updating the summary with {len(frames_)} frames…",
                            eta_llm(0, len(frames_), backend))
                    try:
                        t = time.monotonic()
                        # Same conversation: the transcript is already in context and in the provider's
                        # prompt cache, so only the frames are new.
                        summary = conv.add_frames(frames_)
                        proc.check_cancelled()
                        took("update with frames", t)
                        images = frames_
                    except summarize.SummaryError as e:  # keep the transcript-only summary
                        log.warning("frames follow-up failed: %s", e)
                        notes.append(f"frames follow-up failed: {e}")
        except summarize.SummaryError as e:
            raise PipelineError(str(e), detail=e.detail)
        finally:
            conv.close()  # deletes the CLI session files; they're only needed for the follow-up turn
        # The saved timings are the real work only: a pause shown to one user isn't part of the video's cost.
        total = time.time() - t0 - cpu.waited()
        summary["_stats"] = {"steps": list(timings), "total": total, "llm": llm,
                             "backend": backend, "model": model or conv.model}
        # Keyed by backend + model so users on different models don't overwrite each other's summaries.
        db.save_summary(video.platform, video.video_id, backend, model or conv.model, summary, bool(images))
        result = _finish(video, meta, transcript, source, lang, summary, bool(images), notes, t0)
        if paced:  # the footer shows what this user saw: the lookup, the paced transcript step, the rest
            owed = paced[1] + owed_lookup
            # Videos show their lookup first (paced when it was skipped); files have no lookup step.
            lookup = [("lookup", owed_lookup)] if reuse_meta else [] if is_file else timings[:1]
            result.replay_steps = lookup + [paced] + (timings if reuse_meta or is_file else timings[1:])
            result.replay_total = total + owed
            result.hold = [(st.text(f"🧠 Summarizing with {llm}…"), owed)]
        return result
    finally:
        side.shutdown(wait=True)  # the thumbnail download must be done before its folder goes
        if owns_workdir:
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
            _eta_frames(dur) + eta_llm(0, config.MAX_FRAMES))
    try:
        vid = media.download_video(video, workdir)
        # Phrase matches ("this book", "as you can see") back up the LLM's choice of moments.
        moments_ = times[:frames.MAX_MOMENTS_EACH] + frames.regex_moments(cues)[:frames.MAX_MOMENTS_EACH]
        # A sweep decodes the whole video on every core, so it takes its turn; a few single grabs don't.
        with cpu.slot(st.show, "video", None) if sweep else contextlib.nullcontext():
            images = frames.extract(vid, dur, moments_, workdir, sweep=sweep, sandboxed=video.platform == "file")
        vid.unlink(missing_ok=True)
    except media.Blocked:
        raise
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
    # Check memory before downloading anything: a job that has to wait shouldn't hold a download meanwhile.
    needed = memory.whisper_needs(dur, transcribe.is_loaded())
    if not memory.can_ever_fit(needed):
        raise PipelineError(f"🧠 This video is too long to transcribe on this machine: it would need about "
                            f"{needed / memory.GB:.1f} GB of memory, more than the bot may use.",
                            detail=f"needs {needed} B, cap {memory.cap()} B")
    if not memory.fits_now(needed):
        raise memory.NeedsMemory(needed, dur)  # the worker sets the job aside and retries it later
    whisper = f"Whisper ({config.WHISPER_MODEL}, {config.WHISPER_DEVICE.upper()})"
    st.show(f"🎧 {why} → downloading audio for {whisper}…",
            _eta_audio(dur) + transcribe.estimate(dur) + rest)
    try:
        audio = media.download_audio(video, workdir)
    except media.Blocked:
        raise
    except media.MediaError as e:
        notes.append(f"audio download failed: {e}")
        return None
    # Some downloads (e.g. gallery-dl's h265 TikTok mp4s) have no audio track; Whisper would just fail.
    # A file a user sent was probed in the sandbox; PyAV must not open it in the bot's process.
    has_audio = meta.get("has_audio", True) if video.platform == "file" else media.has_audio_stream(audio)
    if not has_audio:
        notes.append("downloaded file has no audio track")
        return None
    stage = f"🗣 {why} → transcribing {fmt_duration(dur)} of audio with {whisper}…"
    with cpu.slot(st.show, "transcription", transcribe.estimate(dur)):
        # Checked again now that it's this job's turn: no other transcription runs, so the guard is exact.
        if not memory.fits_now(needed):
            raise memory.NeedsMemory(needed, dur)  # the slot is released; the worker parks the job
        estimate = transcribe.estimate(dur) + rest
        st.show(stage, estimate)
        cues, lang, prob = transcribe.transcribe(str(audio), workdir if video.platform == "file" else None,
                                                 whisper_progress(st, stage, estimate, rest))
    if not audio.name.startswith("vid."):  # a TikTok video file stays for frames (deleted with the workdir)
        audio.unlink(missing_ok=True)
    log.info("whisper: %d segments, lang=%s p=%.2f", len(cues), lang, prob)
    st.ok(f"✅ Transcript: Whisper, language {lang}" if cues else "✅ Whisper: no speech found")
    return cues, f"whisper-{config.WHISPER_MODEL}", lang


WHISPER_PROGRESS_EVERY = 10  # seconds between percentage updates (each is a Telegram edit)


def whisper_progress(st: Status, stage: str, estimate: float, rest: float) -> Callable[[float, float], None]:
    """A Whisper progress callback that adds "37 %" to the status line and estimates from this run's rate.

    Updates at most every WHISPER_PROGRESS_EVERY seconds and only when the percentage changed. The voice filter
    skips silence and music, so the percentage can jump; it stays below 100 until Whisper is done. Once 2 % is
    done, the time left is this run's own rate applied to the rest of the audio (plus the later steps),
    instead of the speed measured on earlier runs.

    Args:
        st: The job's status.
        stage: The transcribing line, without the percentage.
        estimate: The estimate shown when transcribing started (counted down until the live rate takes over).
        rest: Estimated seconds for the steps after transcription.
    """
    started = time.monotonic()
    last = {"shown": -1, "at": started}

    def report(done: float, total: float) -> None:
        """Shows the percentage (and a live estimate) for done of total seconds of audio."""
        if total <= 0:
            return
        percent = min(99, int(done / total * 100))
        now = time.monotonic()
        if percent == last["shown"] or now - last["at"] < WHISPER_PROGRESS_EVERY:
            return
        last["shown"], last["at"] = percent, now
        if percent >= 2:
            eta = (now - started) / done * (total - done) + rest
        else:
            eta = max(0.0, estimate - (now - started))
        st.show(f"{stage} {percent} %", eta)

    return report


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
    rest = 0 if transcript_only else eta_llm(_chars_for(dur), 1)
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
    if video.platform == "file":  # a recording someone sent: Whisper is the only source
        if got := _whisper(video, meta, workdir, st, notes, "Transcribing", rest):
            return got
        st.ok("⚠️ No speech found")
        return [], "none", ""

    # TikTok: no transcript API. Audio + whisper first, then TikTok's own auto-captions.
    got = _whisper(video, meta, workdir, st, notes, "TikTok has no transcript", rest)
    if got and got[0]:
        return got
    if caps := media.fetch_captions(video, meta, workdir):
        st.ok(f"✅ Transcript: TikTok captions ({caps[1]})")
        return caps[0], "tiktok-webvtt", caps[1]
    return [], "none", got[2] if got else ""


def cleanup_leftovers() -> int:
    """Removes what a crashed or killed run left behind; call only while no job is running (startup).

    Each job normally deletes its downloads and LLM session files itself, but a kill, out-of-memory or power
    loss skips that: downloads would clutter the disk and session files would keep video content around.

    Returns:
        How many files and folders were removed.
    """
    removed = 0
    targets = list((config.DATA_DIR / "work").glob("*"))      # per-video download folders
    targets += list(config.DATA_DIR.glob("tmp*"))               # LLM job folders (tempfile, prefix "tmp")
    targets += list((config.CODEX_HOME / "sessions").rglob("*.jsonl"))  # Codex conversations
    cwd = summarize.ClaudeCodeConversation.CWD                   # Claude Code names its project dir after it
    targets += list(Path.home().glob(f".claude/projects/{str(cwd).replace('/', '-')}/*.jsonl"))
    for path in targets:
        try:
            shutil.rmtree(path) if path.is_dir() else path.unlink()
            removed += 1
        except OSError as e:
            log.warning("couldn't remove leftover %s: %s", path, e)
    if removed:
        log.info("removed %d leftover file(s)/folder(s) from earlier runs", removed)
    return removed
