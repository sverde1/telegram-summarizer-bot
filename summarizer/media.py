"""Everything that talks to YouTube/TikTok: metadata, captions, audio, video, thumbnail, carousels."""
import json
import logging
import re
import subprocess
import threading
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

from PIL import Image

from . import config, cpu, netjail, proc, sandbox
from .urls import Video

log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


LIVE = ("🔴 Live streams aren't supported. Send the link again once the stream has ended and the recording "
        "is available.")
UPCOMING = "🔴 This stream hasn't started yet. Send the link again once it has ended and the recording is available."


class MediaError(RuntimeError):
    """A download or probe failed; the message is the tool's own error line, fit to show users."""


class NotProcessable(MediaError):
    """The video exists but can't be summarized (e.g. a live stream). The message is shown as-is."""


class Blocked(MediaError):
    """YouTube/TikTok is refusing this server's downloads (rate limit, bot check, IP block).

    Never swallowed by the steps that carry on after a failed download: without it the bot would quietly
    summarize with no transcript.
    """


# Error texts that mean the platform blocks or throttles this server (not a problem with the video itself).
# Note "Sign in to confirm your age" is age-restriction, not a block: "not a bot" is the bot check.
_BLOCKED_PATTERNS = ("http error 429", "too many requests", "not a bot", "ip address is blocked",
                     "rate-limit", "rate limit", "blocked from accessing")


def _is_blocked(message: str) -> bool:
    """Whether a download error means the platform is blocking this server."""
    text = message.lower().replace("’", "'")
    return any(p in text for p in _BLOCKED_PATTERNS)


def describe(e: MediaError) -> str:
    """Turns a download error (yt-dlp's raw last error line) into a message for the user.

    The raw text goes to the admin and the log as detail; users get one of these fixed messages.
    """
    if isinstance(e, NotProcessable):
        return str(e)
    text = str(e).lower()
    if "private" in text:
        return "🔒 This video is private, so the bot can't open it."
    if "age" in text and ("confirm" in text or "sign in" in text or "restricted" in text):
        return "🔞 This video is age-restricted, so the bot can't open it."
    if "country" in text or "geo" in text or "region" in text:
        return "🌍 This video isn't available in the bot's country."
    if "timed out" in text:
        return "⚠️ Downloading the video took too long. Please try again later."
    if any(w in text for w in ("removed", "deleted", "unavailable", "not available", "does not exist", "no longer")):
        return "This video is unavailable (deleted, private or region-locked)."
    return "⚠️ Couldn't load this video. It may be private, deleted or unavailable here."


def _run(cmd: list[str], timeout: int = 900, env: dict | None = None) -> subprocess.CompletedProcess:
    """Runs a command and turns a non-zero exit into a readable MediaError.

    Args:
        cmd: The command and its arguments.
        timeout: Seconds before the command is killed.
        env: Extra environment variables for it (added to config.clean_env()).

    Returns:
        The completed process, with text stdout/stderr.

    Raises:
        MediaError: The command failed. The message is the last "ERROR" line of its output (yt-dlp and
            gallery-dl print warnings and progress around it), or the last line if there is none.
        proc.ProcCancelled: The job was cancelled. Deliberately not turned into MediaError: callers that
            shrug off a failed download must not carry on with a cancelled job.
    """
    log.debug("run: %s", " ".join(cmd))
    try:
        p = proc.run(cmd, timeout=timeout, env=config.clean_env() | env if env else None)
    except proc.ProcTimeout:
        raise MediaError("the download timed out")  # ProcCancelled is deliberately not caught: it stops the job
    # Warnings are kept for YouTube with PO tokens on: a failing token provider is only a warning (yt-dlp then
    # goes on without a token), and it must reach the journal rather than end as a mysterious bot check.
    for line in (p.stderr or "").splitlines():
        if line.startswith("WARNING"):
            log.warning("yt-dlp: %s", line[:500])
    if p.returncode != 0:
        err = (p.stderr or p.stdout).strip().splitlines()
        msg = next((ln for ln in reversed(err) if "ERROR" in ln), err[-1] if err else "unknown error")
        msg = msg.replace("ERROR: ", "")
        if _is_blocked(msg):
            token = next((ln for ln in err if "PO Token" in ln and "WARNING" in ln), "")
            raise Blocked(f"{msg} (PO token: {token[:300]})" if token else msg)
        raise MediaError(msg)
    return p


def _ytdlp(*args: str, timeout: int = 900, youtube: bool = False) -> subprocess.CompletedProcess:
    """Runs the venv's yt-dlp with the options every call shares.

    `config.YTDLP` is the binary next to the interpreter, because an old system yt-dlp on PATH fails on
    current YouTube/TikTok pages. `--sleep-requests 1` keeps us polite to the platforms (one video at a
    time from a personal IP), and `--no-playlist` stops a video URL that is part of a playlist from
    pulling the whole playlist.

    Args:
        *args: yt-dlp arguments, including the URL.
        timeout: Seconds before yt-dlp is killed.
        youtube: It's a YouTube video: PO tokens are used when they're set up (see pot_ready).

    Returns:
        The completed process.

    Raises:
        MediaError: yt-dlp failed.
    """
    if youtube and pot_ready():
        return _run([config.YTDLP, "--no-playlist", "--sleep-requests", "1", *_pot_args(), *args],
                    timeout=timeout, env=_pot_env())
    return _run([config.YTDLP, "--no-warnings", "--no-playlist", "--sleep-requests", "1", *args],
                timeout=timeout)


# ---------- PO tokens ----------

NODE_SANDBOXED = config.ROOT / "deploy" / "node-sandboxed"
_pot: dict = {}  # {"ready": bool, "why": str}, from check_pot at startup


def _pot_args() -> list[str]:
    """yt-dlp arguments for PO tokens: the plugin's script mode, and Node only through the sandbox wrapper.

    `--js-runtimes` adds node next to yt-dlp's default deno (its signature solver keeps preferring deno); one
    `--extractor-args` per extractor, since `;` separates arguments of the same one.
    """
    return ["--extractor-args", f"youtubepot-bgutilscript:server_home={config.POT_HOME}",
            "--js-runtimes", f"node:{NODE_SANDBOXED}"]


def _pot_env() -> dict:
    """What deploy/node-sandboxed needs from the bot: the script's directory and its token cache."""
    return {"POT_SERVER_HOME": str(config.POT_HOME), "POT_CACHE": str(config.POT_HOME.parent / "cache")}


def pot_ready() -> bool:
    """Whether YouTube calls use PO tokens (decided once by check_pot at startup)."""
    return _pot.get("ready", False)


def check_pot() -> str:
    """Checks the PO-token setup once (at startup): the plugin, the built script, the sandboxed Node.

    Returns:
        "" when PO tokens are on, else why they're off.
    """
    from importlib.metadata import PackageNotFoundError, version
    try:
        plugin = version("bgutil-ytdlp-pot-provider")
    except PackageNotFoundError:
        plugin = ""
    script = config.POT_HOME / "build" / "generate_once.js"
    why = ""
    if not plugin and not script.exists():
        why = "not installed (deploy/install-pot.sh)"
    elif not plugin or not script.exists():
        why = "half installed: rerun deploy/install-pot.sh"
    elif (config.POT_HOME / "src").exists():
        # The plugin would prefer its deno variant from src/, which runs outside the sandbox.
        why = f"{config.POT_HOME / 'src'} must not exist: rerun deploy/install-pot.sh"
    elif not netjail.ready():
        why = "the network jail doesn't work (see its startup line): the token script runs only in it"
    else:
        try:
            p = proc.run([str(NODE_SANDBOXED), str(script), "--version"], timeout=30,
                         env=config.clean_env() | _pot_env())
            got = p.stdout.strip()
            if p.returncode != 0 or got != plugin:
                why = f"the token script says {got or p.stderr.strip()[:200]!r}, the plugin is {plugin}"
        except (OSError, proc.ProcError) as e:
            why = f"the sandboxed Node failed: {e}"
    _pot.update(ready=not why, why=why, version=plugin)
    return why


_youtube = threading.BoundedSemaphore(config.YOUTUBE_PARALLEL)  # see _youtube_turn
_turn_held = threading.local()  # whether this thread holds a turn already


@contextmanager
def _youtube_turn(video: Video):
    """Waits for one of the YOUTUBE_PARALLEL turns before a request to youtube.com itself (cancellably).

    Lookups and caption requests are what YouTube's bot check counts; a burst of them from parallel jobs gets
    the IP flagged. Downloads from a looked-up video (googlevideo) don't take a turn: a long one mustn't hold
    up everyone's lookups. The wait is left out of the job's step times (cpu.record_wait). Other platforms
    pass straight through.

    Raises:
        proc.ProcCancelled: The job was cancelled while waiting.
    """
    # Re-entrant per thread: captions take a turn and their fallback lookup takes it again; with one turn,
    # waiting for itself would hang the job.
    if video.platform != "youtube" or getattr(_turn_held, "on", False):
        yield
        return
    start = time.monotonic()
    while not _youtube.acquire(timeout=1):
        proc.check_cancelled()
    cpu.record_wait(start, time.monotonic())
    _turn_held.on = True
    try:
        yield
    finally:
        _turn_held.on = False
        _youtube.release()


INFO_JSON = "info.json"  # yt-dlp's full answer from the probe, in the job's workdir (see _from_info)
# yt-dlp errors that mean the stored info's media URLs have expired (YouTube signs them for a few hours).
_EXPIRED = re.compile(r"\b(403|410)\b|forbidden|expired", re.I)


def _from_info(video: Video, workdir: Path, args: list[str], timeout: int) -> subprocess.CompletedProcess:
    """Runs yt-dlp for this job's video, from the probe's stored info when there is one.

    `--load-info-json` makes yt-dlp skip extraction: no second (third, fourth) load of the video page, no
    `--sleep-requests` waits, and the same formats and subtitles the probe saw. If the stored URLs have
    expired, the video is looked up again once. Without stored info (e.g. a summary written from a cached
    transcript, where the probe was skipped) the URL is used, which does the lookup then.

    Raises:
        MediaError: yt-dlp failed (Blocked for a platform ban, never retried).
    """
    info = workdir / INFO_JSON
    if info.exists():
        try:
            return _ytdlp(*args, "--load-info-json", str(info), timeout=timeout,
                          youtube=video.platform == "youtube")
        except Blocked:
            raise
        except MediaError as e:
            if not _EXPIRED.search(str(e)):
                raise
            log.info("stored video info expired (%s); looking the video up again", e)
            info.unlink(missing_ok=True)
    with _youtube_turn(video):  # without stored info, this looks the video up
        return _ytdlp(*args, video.url, timeout=timeout, youtube=video.platform == "youtube")


# ---------- metadata ----------

def probe(video: Video, info_path: Path | None = None) -> dict:
    """Fetches a video's metadata in one cheap request, without downloading media.

    Args:
        video: The classified video.
        info_path: Where to keep yt-dlp's full answer, so this job's later yt-dlp calls don't look the
            video up again (see _from_info). Job workdir only: it holds signed URLs and cookies.

    Returns:
        A dict with id, title, description (first 3000 characters), uploader, duration (seconds, 0 if
        unknown), upload_date, language, thumbnail URL, subtitles (manual tracks by language),
        auto_captions (language codes), is_carousel and music (track name).

    Raises:
        MediaError: The video is unavailable (deleted, private, region-locked) or yt-dlp failed.
    """
    # A TikTok carousel without music has no formats at all; still return its metadata.
    extra = ["--ignore-no-formats-error"] if video.platform == "tiktok" else []
    try:
        with _youtube_turn(video):
            raw = _ytdlp("--skip-download", "-J", *extra, video.url, timeout=120,
                         youtube=video.platform == "youtube").stdout
        d = json.loads(raw)
    except MediaError as e:
        # yt-dlp often refuses live streams itself ("This live stream recording is not available.", "This
        # live event will begin in …", "Premieres in …"): give those the same clear message.
        text = str(e).lower()
        if "live event will begin" in text or "premieres in" in text:
            raise NotProcessable(UPCOMING)
        if "live stream" in text or "is live" in text:
            raise NotProcessable(LIVE)
        raise
    # A live stream has no end: downloading it would block the queue until the timeout and keep writing to
    # disk. "post_live" is a just-ended stream whose recording isn't processed yet; "was_live" (a finished
    # stream with a recording) is fine.
    live = d.get("live_status")
    if live in ("is_live", "post_live") or d.get("is_live"):
        raise NotProcessable(LIVE)
    if live == "is_upcoming":
        raise NotProcessable(UPCOMING)
    if info_path is not None:
        info_path.write_text(raw)
    title = d.get("title") or ""
    if video.platform == "tiktok":
        # TikTok "title" is a truncated description; the full caption is more useful.
        title = d.get("description") or title
    return {
        "id": d.get("id"),
        "title": title.strip(),
        "description": (d.get("description") or "")[:3000],
        "uploader": d.get("uploader") or d.get("channel") or "",
        # An id, unlike the display name: whether several links are one creator's (parts of one video).
        "uploader_id": d.get("channel_id") or d.get("uploader_id") or d.get("uploader") or "",
        "timestamp": d.get("timestamp") or 0,
        "duration": d.get("duration") or 0,
        "upload_date": d.get("upload_date") or "",
        "language": d.get("language") or "",
        "thumbnail": d.get("thumbnail") or "",
        # live_chat is YouTube's chat replay of a stream, exposed as a "subtitle" track; not a transcript.
        "subtitles": {k: v for k, v in (d.get("subtitles") or {}).items() if k != "live_chat"},
        "auto_captions": sorted((d.get("automatic_captions") or {}).keys()),
        # TikTok photo carousel: yt-dlp only sees the background music (no video formats).
        "is_carousel": video.platform == "tiktok" and (video.kind == "photo" or all(
            (f.get("vcodec") or "none") == "none" for f in d.get("formats") or [])),
        "music": d.get("track") or "",
    }


# ---------- captions ----------

# VTT timestamps come as HH:MM:SS.mmm or, for short files, MM:SS.mmm (some writers use a comma).
_TS = re.compile(r"(\d+):(\d\d):(\d\d)[.,](\d+)|(\d+):(\d\d)[.,](\d+)")


def _ts_seconds(s: str) -> float:
    """Converts the first VTT timestamp in `s` to seconds.

    Args:
        s: Text containing a timestamp such as "01:02:03.500" or "02:03.500".

    Returns:
        The time in seconds, or 0.0 if `s` contains no timestamp.
    """
    m = _TS.search(s)
    if not m:
        return 0.0
    if m.group(1):
        h, mi, se, ms = m.group(1, 2, 3, 4)
    else:
        h, (mi, se, ms) = "0", m.group(5, 6, 7)
    return int(h) * 3600 + int(mi) * 60 + int(se) + int(ms) / 10 ** len(ms)


def parse_vtt(vtt: str) -> list[tuple[float, str]]:
    """Parses WebVTT captions into timed lines of plain text.

    YouTube's auto-captions are "rolling": each cue repeats the previous line before adding a new one,
    so consecutive duplicate lines are dropped. Inline tags (word timings, styling) are stripped.

    Args:
        vtt: The contents of a .vtt file.

    Returns:
        [(start_seconds, text)], one entry per distinct line, in order. The start is that of the cue
        where the line first appears; the timestamps let the LLM name moments to look at.
    """
    cues, start, prev = [], 0.0, None
    for ln in vtt.splitlines():
        ln = ln.strip()
        if "-->" in ln:
            start = _ts_seconds(ln.split("-->")[0])
            continue
        if not ln or ln.startswith(("WEBVTT", "Kind:", "Language:", "NOTE")) or ln.isdigit():
            continue
        ln = re.sub(r"<[^>]+>", "", ln).strip()
        if ln and ln != prev:
            cues.append((start, ln))
            prev = ln
    return cues


def _pick_caption_lang(meta: dict) -> tuple[str, bool] | None:
    """Chooses ONE caption track to download.

    One track means one request: YouTube rate-limits and IP-bans repeated caption fetches. Manual
    subtitles win over auto-captions (they're accurate); the video's own language wins over English
    (the summarizer translates, and a translated track loses detail).

    Args:
        meta: Metadata from `probe`.

    Returns:
        (language code, is_auto_caption), or None if the video has no captions at all.
    """
    lang = (meta.get("language") or "").split("-")[0]
    manual = list(meta["subtitles"])
    for want in (lang, "en"):
        if want:
            for k in manual:
                if k == want or k.startswith(want + "-"):
                    return k, False
    if manual:
        return manual[0], False
    auto = meta["auto_captions"]
    # "<lang>-orig" is the untranslated auto-caption track in the spoken language; the plain codes are
    # usually machine translations of it into every language YouTube offers.
    for k in [f"{lang}-orig", lang, "en-orig", "en"] + [a for a in auto if a.endswith("-orig")]:
        if k and k in auto:
            return k, True
    return None


def fetch_captions(video: Video, meta: dict, workdir: Path) -> tuple[list[tuple[float, str]], str] | None:
    """Downloads the platform's captions for a video, if it has any.

    Used for YouTube first, and for TikTok's own auto-captions when Whisper fails. A failed download is
    logged and treated as "no captions" so the caller can fall back to Whisper.

    Args:
        video: The classified video.
        meta: Metadata from `probe`.
        workdir: The job's temp directory; the .vtt file is written there.

    Returns:
        (cues, language code without the "-orig" suffix), or None if there are no usable captions.
    """
    choice = _pick_caption_lang(meta)
    if not choice:
        return None
    lang, is_auto = choice
    flag = "--write-auto-subs" if is_auto else "--write-subs"
    try:
        with _youtube_turn(video):  # timedtext is on youtube.com, and it's rate limited
            _from_info(video, workdir, ["--skip-download", flag, "--sub-langs", lang, "--sub-format", "vtt/best",
                                        "-o", str(workdir / "cap.%(ext)s")], timeout=120)
    except Blocked:
        raise  # a ban affects every download: report it instead of carrying on without captions
    except MediaError as e:
        log.warning("caption download failed: %s", e)
        return None
    files = sorted(workdir.glob("cap.*.vtt"))
    if not files:
        return None
    cues = parse_vtt(files[0].read_text(encoding="utf-8", errors="replace"))
    return (cues, lang.removesuffix("-orig")) if cues else None


# ---------- audio / video ----------

def download_audio(video: Video, workdir: Path) -> Path:
    """Downloads a video's audio track for Whisper.

    Args:
        video: The classified video.
        workdir: The job's temp directory.

    Returns:
        Path of the audio file, in whatever container the platform serves (m4a, webm, mp3...). For TikTok
        it is the video file itself (see below), which frames then reuse.

    Raises:
        MediaError: yt-dlp failed or produced no file.
    """
    if video.platform == "file":  # a file a user sent is already in the job's directory
        return _sent_file(workdir, ("audio.*", "vid.*"))
    if video.platform == "tiktok":
        # TikTok has no separate audio stream: "bestaudio" was the whole video anyway, and frames downloaded
        # it a second time. One download, of the format frames want, serves both.
        return download_video(video, workdir)
    # No re-encode (no -x): transcribe decodes any container with ffmpeg itself, and yt-dlp's audio
    # extraction would need ffprobe, which the bundled imageio-ffmpeg doesn't include.
    _from_info(video, workdir, ["-f", "bestaudio/best", "-o", str(workdir / "audio.%(ext)s")], timeout=900)
    # A .part file is an interrupted download, not the result.
    files = [f for f in workdir.glob("audio.*") if f.suffix != ".part"]
    if not files:
        raise MediaError("audio download produced no file")
    return files[0]


def download_video(video: Video, workdir: Path) -> Path:
    """Downloads a playable video file for frame grabbing.

    Args:
        video: The classified video.
        workdir: The job's temp directory.

    Returns:
        Path of the video file (one already downloaded in this job is reused).

    Raises:
        MediaError: yt-dlp failed or produced no file.
    """
    if existing := [f for f in workdir.glob("vid.*") if f.suffix != ".part"]:
        return existing[0]  # TikTok: already downloaded for Whisper; a video file a user sent
    if video.platform == "file":
        raise MediaError("the file has no video")
    if video.platform == "tiktok":
        # An h264 stream (decodes everywhere; some TikTok formats are h265), else whatever exists. Not
        # "download": in yt-dlp's TikTok extractor that is the watermarked web file.
        fmt = "best[vcodec^=h264]/best"
    else:
        # 720p is plenty to read on-screen text, and keeps long YouTube downloads small. A combined file
        # (b) first, else video-only (bv*: frames don't need audio, and nothing has to be merged);
        # h264 (avc1) first within each, since it decodes everywhere.
        fmt = "b[height<=720][vcodec^=avc1]/bv*[height<=720][vcodec^=avc1]/b[height<=720]/bv*[height<=720]/b"
    _from_info(video, workdir, ["-f", fmt, "-o", str(workdir / "vid.%(ext)s")], timeout=1800)
    files = [f for f in workdir.glob("vid.*") if f.suffix != ".part"]
    if not files:
        raise MediaError("video download produced no file")
    return files[0]


def _sent_file(workdir: Path, patterns: tuple[str, ...]) -> Path:
    """The file a user sent, in the job's directory (named audio.* or vid.* by the bot).

    Raises:
        MediaError: It isn't there.
    """
    for pattern in patterns:
        if found := [f for f in workdir.glob(pattern) if f.suffix != ".part"]:
            return found[0]
    raise MediaError("the file is missing")


COVER_CODECS = {"mjpeg", "png", "bmp", "gif", "webp"}  # a still image in an audio file is cover art


def probe_file(path: Path, workdir: Path) -> dict:
    """Length and streams of a file a user sent, read in the sandbox (the file is untrusted).

    Cover art (an attached picture, or a single still image) doesn't count as video: an mp3 with a cover is
    audio.

    Args:
        path: The file, inside workdir.
        workdir: The job's directory (the sandbox's /job).

    Returns:
        {"duration": seconds (0 if unknown), "has_audio": bool, "has_video": bool}.

    Raises:
        MediaError: The file can't be read as audio or video.
    """
    target = sandbox.inside(path, workdir)
    if config.FFPROBE:
        p = proc.run(sandbox.command(workdir, [config.FFPROBE, "-v", "error", "-show_format", "-show_streams",
                                               "-of", "json", target], memory=2 * 1024 ** 3), timeout=120)
        try:
            info = json.loads(p.stdout or "{}")
        except ValueError:
            info = {}
        streams = info.get("streams") or []
        if p.returncode != 0 or not streams:
            raise MediaError(f"unreadable media: {p.stderr[-200:]}")
        has_audio = any(s.get("codec_type") == "audio" for s in streams)
        has_video = any(s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")
                        and not (s.get("codec_name") in COVER_CODECS and str(s.get("nb_frames", "1")) in ("0", "1"))
                        for s in streams)
        try:
            duration = float((info.get("format") or {}).get("duration") or 0)
        except ValueError:
            duration = 0.0
        return {"duration": duration, "has_audio": has_audio, "has_video": has_video}
    # No ffprobe (the imageio ffmpeg build): "ffmpeg -i" lists the streams and fails for want of an output.
    p = proc.run(sandbox.command(workdir, [config.FFMPEG, "-hide_banner", "-i", target], memory=2 * 1024 ** 3),
                 timeout=120)
    text = p.stderr or ""
    streams = re.findall(r"Stream #\S+.*?: (Audio|Video): (\w+)(.*)", text)
    if not streams:
        raise MediaError("unreadable media")
    m = re.search(r"Duration: (\d+):(\d\d):(\d\d(?:\.\d+)?)", text)
    duration = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]) if m else 0.0
    has_video = any(kind == "Video" and not (codec in COVER_CODECS or "attached pic" in rest)
                    for kind, codec, rest in streams)
    return {"duration": duration, "has_audio": any(k == "Audio" for k, _, _ in streams), "has_video": has_video}


def has_audio_stream(path: Path) -> bool:
    """Checks whether a media file contains an audio stream.

    Some TikTok downloads are video-only; sending those to Whisper would just produce an empty
    transcript after a wasted decode.

    Args:
        path: The media file.

    Returns:
        True if the file has at least one audio stream; False if it has none or can't be opened.
    """
    # PyAV is only used to list streams (faster-whisper already depends on it); decoding goes via ffmpeg.
    import av
    try:
        with av.open(str(path)) as c:
            return any(s.type == "audio" for s in c.streams)
    except Exception:
        return False


# ---------- images ----------

def to_jpeg(src: Path, dst: Path, max_edge: int = 1568) -> Path:
    """Converts an image to an RGB JPEG no larger than `max_edge` on its long side.

    Thumbnails and slides arrive as webp/png/jpeg; the LLM backends take JPEG. 1568 px is the long edge
    beyond which vision models gain little, while images cost more tokens and upload time.

    Args:
        src: The source image.
        dst: Where to write the JPEG.
        max_edge: Maximum width and height in pixels; aspect ratio is kept, smaller images are not enlarged.

    Returns:
        `dst`.
    """
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((max_edge, max_edge))
        im.save(dst, "JPEG", quality=88)
    return dst


def download_thumbnail(meta: dict, workdir: Path) -> Path | None:
    """Downloads the video's thumbnail as a JPEG (for the clickbait check).

    The thumbnail is optional: on any failure the summary is written without it.

    Args:
        meta: Metadata from `probe`.
        workdir: The job's temp directory.

    Returns:
        Path of thumb.jpg, or None if there is no thumbnail or it couldn't be downloaded/decoded.
    """
    url = meta.get("thumbnail")
    if not url:
        return None
    raw = workdir / "thumb.raw"
    try:
        # A browser User-Agent: some image CDNs reject Python's default one.
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            raw.write_bytes(r.read())
        return to_jpeg(raw, workdir / "thumb.jpg")
    except Exception as e:
        log.warning("thumbnail failed: %s", e)
        return None
    finally:
        raw.unlink(missing_ok=True)


def download_carousel(video: Video, workdir: Path) -> list[Path]:
    """Downloads the slides of a TikTok photo post: the images are the content.

    yt-dlp only sees a carousel's background music, so gallery-dl fetches the images. A gallery-dl
    failure is logged and returns whatever was downloaded (possibly nothing); the caller reports an
    empty result to the user.

    Args:
        video: The classified video (TikTok).
        workdir: The job's temp directory; slides go to workdir/slides.

    Returns:
        The slide images in order (named 01, 02, ... by gallery-dl's post order).
    """
    out = workdir / "slides"
    out.mkdir(exist_ok=True)
    try:
        # Zero-padded names so sorting by name keeps the slide order past 9 slides.
        _run([config.GALLERY_DL, "--sleep-request", "1", "-D", str(out),
              "-f", "{num:>02}.{extension}", video.url], timeout=300)  # also fetches the music; dropped below
    except Blocked:
        raise
    except MediaError as e:
        log.warning("gallery-dl failed: %s", e)
    imgs = sorted(p for p in out.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    for p in out.iterdir():  # gallery-dl may also fetch the music track; we don't need it
        if p not in imgs:
            p.unlink()
    return imgs
