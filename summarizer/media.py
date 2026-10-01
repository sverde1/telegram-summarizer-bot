"""Everything that talks to YouTube/TikTok: metadata, captions, audio, video, thumbnail, carousels."""
import json
import logging
import re
import subprocess
import urllib.request
from pathlib import Path

from PIL import Image

from . import config, proc
from .urls import Video

log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


class MediaError(RuntimeError):
    """A download or probe failed; the message is the tool's own error line, fit to show users."""


def _run(cmd: list[str], timeout: int = 900) -> subprocess.CompletedProcess:
    """Runs a command and turns a non-zero exit into a readable MediaError.

    Args:
        cmd: The command and its arguments.
        timeout: Seconds before the command is killed.

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
        p = proc.run(cmd, timeout=timeout)
    except proc.ProcTimeout:
        raise MediaError("the download timed out")  # ProcCancelled is deliberately not caught: it stops the job
    if p.returncode != 0:
        err = (p.stderr or p.stdout).strip().splitlines()
        msg = next((ln for ln in reversed(err) if "ERROR" in ln), err[-1] if err else "unknown error")
        raise MediaError(msg.replace("ERROR: ", ""))
    return p


def _ytdlp(*args: str, timeout: int = 900) -> subprocess.CompletedProcess:
    """Runs the venv's yt-dlp with the options every call shares.

    `config.YTDLP` is the binary next to the interpreter, because an old system yt-dlp on PATH fails on
    current YouTube/TikTok pages. `--sleep-requests 1` keeps us polite to the platforms (one video at a
    time from a personal IP), and `--no-playlist` stops a video URL that is part of a playlist from
    pulling the whole playlist.

    Args:
        *args: yt-dlp arguments, including the URL.
        timeout: Seconds before yt-dlp is killed.

    Returns:
        The completed process.

    Raises:
        MediaError: yt-dlp failed.
    """
    return _run([config.YTDLP, "--no-warnings", "--no-playlist", "--sleep-requests", "1", *args],
                timeout=timeout)


# ---------- metadata ----------

def probe(video: Video) -> dict:
    """Fetches a video's metadata in one cheap request, without downloading media.

    Args:
        video: The classified video.

    Returns:
        A dict with id, title, description (first 3000 characters), uploader, duration (seconds, 0 if
        unknown), upload_date, language, thumbnail URL, subtitles (manual tracks by language),
        auto_captions (language codes), is_carousel and music (track name).

    Raises:
        MediaError: The video is unavailable (deleted, private, region-locked) or yt-dlp failed.
    """
    # A TikTok carousel without music has no formats at all; still return its metadata.
    extra = ["--ignore-no-formats-error"] if video.platform == "tiktok" else []
    d = json.loads(_ytdlp("--skip-download", "-J", *extra, video.url, timeout=120).stdout)
    title = d.get("title") or ""
    if video.platform == "tiktok":
        # TikTok "title" is a truncated description; the full caption is more useful.
        title = d.get("description") or title
    return {
        "id": d.get("id"),
        "title": title.strip(),
        "description": (d.get("description") or "")[:3000],
        "uploader": d.get("uploader") or d.get("channel") or "",
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
        _ytdlp("--skip-download", flag, "--sub-langs", lang, "--sub-format", "vtt/best",
               "-o", str(workdir / "cap.%(ext)s"), video.url, timeout=120)
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
        Path of the audio file, in whatever container the platform serves (m4a, webm, mp3...).

    Raises:
        MediaError: yt-dlp failed or produced no file.
    """
    # No re-encode (no -x): transcribe decodes any container with ffmpeg itself, and yt-dlp's audio
    # extraction would need ffprobe, which the bundled imageio-ffmpeg doesn't include.
    _ytdlp("-f", "bestaudio/best", "-o", str(workdir / "audio.%(ext)s"), video.url)
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
        Path of the video file.

    Raises:
        MediaError: yt-dlp failed or produced no file.
    """
    if video.platform == "tiktok":
        # "download" is TikTok's watermark-free format; then any h264 stream (decodes everywhere; some
        # TikTok formats are h265), then whatever exists.
        fmt = "download/best[vcodec^=h264]/best"
    else:
        # 720p is plenty to read on-screen text, and keeps long YouTube downloads small. A combined file
        # (b) first, else video-only (bv*: frames don't need audio, and nothing has to be merged);
        # h264 (avc1) first within each, since it decodes everywhere.
        fmt = "b[height<=720][vcodec^=avc1]/bv*[height<=720][vcodec^=avc1]/b[height<=720]/bv*[height<=720]/b"
    _ytdlp("-f", fmt, "-o", str(workdir / "vid.%(ext)s"), video.url, timeout=1800)
    files = [f for f in workdir.glob("vid.*") if f.suffix != ".part"]
    if not files:
        raise MediaError("video download produced no file")
    return files[0]


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
    except MediaError as e:
        log.warning("gallery-dl failed: %s", e)
    imgs = sorted(p for p in out.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    for p in out.iterdir():  # gallery-dl may also fetch the music track; we don't need it
        if p not in imgs:
            p.unlink()
    return imgs
