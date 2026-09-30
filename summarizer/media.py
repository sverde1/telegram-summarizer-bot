"""Everything that talks to YouTube/TikTok: metadata, captions, audio, video, thumbnail, carousels."""
import json
import logging
import re
import subprocess
import urllib.request
from pathlib import Path

from PIL import Image

from . import config
from .urls import Video

log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


class MediaError(RuntimeError):
    pass


def _run(cmd: list[str], timeout: int = 900) -> subprocess.CompletedProcess:
    log.debug("run: %s", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if p.returncode != 0:
        err = (p.stderr or p.stdout).strip().splitlines()
        msg = next((ln for ln in reversed(err) if "ERROR" in ln), err[-1] if err else "unknown error")
        raise MediaError(msg.replace("ERROR: ", ""))
    return p


def _ytdlp(*args: str, timeout: int = 900) -> subprocess.CompletedProcess:
    return _run([config.YTDLP, "--no-warnings", "--no-playlist", "--sleep-requests", "1", *args],
                timeout=timeout)


# ---------- metadata ----------

def probe(video: Video) -> dict:
    """One cheap request: title, uploader, duration, thumbnail, available captions."""
    d = json.loads(_ytdlp("--skip-download", "-J", video.url, timeout=120).stdout)
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
        "subtitles": {k: v for k, v in (d.get("subtitles") or {}).items() if k != "live_chat"},
        "auto_captions": sorted((d.get("automatic_captions") or {}).keys()),
        "image_post": bool(d.get("_type") == "playlist" or video.kind == "photo"),
    }


# ---------- captions ----------

_TS = re.compile(r"(\d+):(\d\d):(\d\d)[.,](\d+)|(\d+):(\d\d)[.,](\d+)")


def _ts_seconds(s: str) -> float:
    m = _TS.search(s)
    if not m:
        return 0.0
    if m.group(1):
        h, mi, se, ms = m.group(1, 2, 3, 4)
    else:
        h, (mi, se, ms) = "0", m.group(5, 6, 7)
    return int(h) * 3600 + int(mi) * 60 + int(se) + int(ms) / 10 ** len(ms)


def parse_vtt(vtt: str) -> list[tuple[float, str]]:
    """VTT -> [(start_seconds, text)]. Auto-captions repeat lines as rolling cues; drop repeats."""
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
    """Choose ONE caption track (fewer requests = less rate limiting). Returns (lang, is_auto)."""
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
    # "<lang>-orig" is the untranslated auto-caption track in the spoken language.
    for k in [f"{lang}-orig", lang, "en-orig", "en"] + [a for a in auto if a.endswith("-orig")]:
        if k and k in auto:
            return k, True
    return None


def fetch_captions(video: Video, meta: dict, workdir: Path) -> tuple[list[tuple[float, str]], str] | None:
    """Returns (cues, language) from platform captions, or None if there are none."""
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
    # No re-encode: faster-whisper decodes m4a/webm/mp3 itself (PyAV), so ffprobe isn't needed.
    _ytdlp("-f", "bestaudio/best", "-o", str(workdir / "audio.%(ext)s"), video.url)
    files = [f for f in workdir.glob("audio.*") if f.suffix != ".part"]
    if not files:
        raise MediaError("audio download produced no file")
    return files[0]


def download_video(video: Video, workdir: Path) -> Path:
    if video.platform == "tiktok":
        fmt = "download/best[vcodec^=h264]/best"
    else:
        fmt = "b[height<=720][vcodec^=avc1]/bv*[height<=720][vcodec^=avc1]/b[height<=720]/bv*[height<=720]/b"
    _ytdlp("-f", fmt, "-o", str(workdir / "vid.%(ext)s"), video.url, timeout=1800)
    files = [f for f in workdir.glob("vid.*") if f.suffix != ".part"]
    if not files:
        raise MediaError("video download produced no file")
    return files[0]


def has_audio_stream(path: Path) -> bool:
    import av
    try:
        with av.open(str(path)) as c:
            return any(s.type == "audio" for s in c.streams)
    except Exception:
        return False


# ---------- images ----------

def to_jpeg(src: Path, dst: Path, max_edge: int = 1568) -> Path:
    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail((max_edge, max_edge))
        im.save(dst, "JPEG", quality=88)
    return dst


def download_thumbnail(meta: dict, workdir: Path) -> Path | None:
    url = meta.get("thumbnail")
    if not url:
        return None
    raw = workdir / "thumb.raw"
    try:
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
    """TikTok photo post: the images are the content."""
    out = workdir / "slides"
    out.mkdir(exist_ok=True)
    try:
        _run([config.GALLERY_DL, "--sleep-request", "1", "-D", str(out),
              "-f", "{num:>02}.{extension}", video.url], timeout=300)
    except MediaError as e:
        log.warning("gallery-dl failed: %s", e)
    imgs = sorted(p for p in out.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    for p in out.iterdir():  # gallery-dl may also fetch the music track; we don't need it
        if p not in imgs:
            p.unlink()
    return imgs
