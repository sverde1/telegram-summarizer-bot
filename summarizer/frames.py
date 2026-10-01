"""Pulling a small set of distinct frames at the moments the LLM asked for (plus an even sweep)."""
import logging
import re
import subprocess
from pathlib import Path

import imagehash
from PIL import Image

from . import config

log = logging.getLogger(__name__)

# Phrases where a speaker points at something on screen. A backup for the LLM's own choice of moments
# (it missed "this book" moments in testing); a few non-English phrases because the videos aren't all
# in English.
SCREEN_REF = re.compile(
    r"as you can see|on (the )?screen|here are the (full )?results|this (table|chart|graph|book|app|list)|"
    r"these (books|apps|tools)|look at (this|the)|the chart|shown here|right here|check (this|it) out|"
    r"link in (the )?(description|bio)|hier (siehst|sieht)|auf dem (chart|bild)|"
    r"comme vous (pouvez )?voir|como (puedes|pueden) ver",
    re.I)


def is_short(meta: dict) -> bool:
    """Tells whether a video is short enough for a dense frame sweep (TikToks, Shorts).

    Args:
        meta: Metadata from `media.probe`.

    Returns:
        True if the duration is at most `config.SHORT_VIDEO_SEC` (unknown duration counts as short).
    """
    return (meta.get("duration") or 0) <= config.SHORT_VIDEO_SEC


def regex_moments(cues: list[tuple[float, str]]) -> list[float]:
    """Finds transcript moments where the speaker refers to something on screen.

    Args:
        cues: [(start_seconds, text)] from captions or Whisper.

    Returns:
        Start times of the cues matching `SCREEN_REF`.
    """
    return [start for start, text in cues if SCREEN_REF.search(text)]


def _mmss(t: float) -> str:
    """Formats seconds as m:ss, or h:mm:ss from one hour on (frame labels for the LLM)."""
    t = int(t)
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


def _grab(video: Path, t: float, out: Path) -> bool:
    """Saves the single frame at `t` seconds as a JPEG.

    Args:
        video: The video file.
        t: Time in seconds.
        out: Where to write the JPEG.

    Returns:
        True if a non-empty image was written (False e.g. past the end of the video).
    """
    # -ss before -i seeks on keyframes first: fast even deep into a long video. Width is capped at
    # 1280 px: enough to read on-screen text, smaller to send; -2 keeps the aspect ratio (even height).
    p = subprocess.run([config.FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{t:.2f}",
                        "-i", str(video), "-frames:v", "1", "-vf", "scale='min(1280,iw)':-2",
                        "-q:v", "2", str(out)], capture_output=True, timeout=120)
    return p.returncode == 0 and out.exists() and out.stat().st_size > 0


def _sweep(video: Path, interval: float, fdir: Path) -> list[tuple[float, Path]]:
    """Grabs a frame every `interval` seconds in a single ffmpeg pass.

    One pass decodes the video once, which is much faster than seeking for each frame.

    Args:
        video: The video file.
        interval: Seconds between frames.
        fdir: Directory for the frames (s_0001.jpg, ...).

    Returns:
        [(approximate time in seconds, path)] in order. The fps filter's first frame is at 0 s.
    """
    subprocess.run([config.FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
                    "-vf", f"fps=1/{interval:.3f},scale='min(1280,iw)':-2", "-q:v", "2",
                    str(fdir / "s_%04d.jpg")], capture_output=True, timeout=900)
    return [(i * interval, p) for i, p in enumerate(sorted(fdir.glob("s_*.jpg")))]


def _spread(items: list, n: int) -> list:
    """Picks `n` items evenly spread over a list, keeping the first and last.

    Used instead of truncating, so a capped selection still covers the whole video (truncating cut off
    the end of long videos).

    Args:
        items: Items in time order.
        n: How many to keep.

    Returns:
        `items` unchanged if it has at most `n` items, otherwise `n` evenly spaced ones.
    """
    if len(items) <= n:
        return items
    if n <= 1:
        return items[:n]
    return [items[round(i * (len(items) - 1) / (n - 1))] for i in range(n)]


def extract(video: Path, duration: float, moments: list[float], workdir: Path,
            sweep: bool) -> list[tuple[Path, str]]:
    """Grabs distinct frames at the given moments, plus an optional even sweep.

    Near-duplicates are dropped, and at most `config.MAX_FRAMES` are kept, preferring the targeted ones.

    Args:
        video: The video file.
        duration: Video length in seconds (moments past the end are skipped).
        moments: Times in seconds where the transcript refers to something shown. Frames are taken
            1 s and 3 s after each: the thing is usually shown just after it's mentioned, and two grabs
            catch it whether it appears quickly or after a cut.
        workdir: The job's temp directory; frames go to workdir/frames.
        sweep: Also sample the whole video evenly (short or speechless videos, or when the LLM asked to
            see the video without naming moments).

    Returns:
        [(path, label)] in time order, the label being "t=m:ss" for the LLM.
    """
    duration = max(duration, 1)
    fdir = workdir / "frames"
    fdir.mkdir(exist_ok=True)

    targeted: list[tuple[float, Path]] = []
    for m in sorted(set(moments)):
        for t in (m + 1, m + 3):
            out = fdir / f"t_{t:08.2f}.jpg"
            if t < duration and _grab(video, t, out):
                targeted.append((t, out))
    # About 40 frames over the whole video, but never denser than every 2 s (a 10-second clip would
    # otherwise give frames 0.25 s apart, all identical) or sparser than every 20 s.
    swept = _sweep(video, min(max(duration / 40, 2), 20), fdir) if sweep else []

    def dedup(frames, against):
        """Drops frames that look like an already kept one.

        Args:
            frames: [(time, path)] candidates, in order.
            against: Already kept [(time, path, hash)] to compare with as well.

        Returns:
            [(time, path, hash)] of the frames kept.
        """
        kept = []
        for t, p in frames:
            with Image.open(p) as im:
                h = imagehash.phash(im)
            # Perceptual-hash distance <= 5 of 64 bits: the same shot with minor motion or compression
            # differences, while a new slide, overlay or table changes many more bits.
            if any(h - k[2] <= 5 for k in against + kept):  # talking-head stretches collapse to one
                continue
            kept.append((t, p, h))
        return kept

    # Targeted frames first, but leave a third of the budget for the sweep when there is one.
    tk = _spread(dedup(targeted, []), config.MAX_FRAMES * 2 // 3 if swept else config.MAX_FRAMES)
    sk = _spread(dedup(swept, tk), config.MAX_FRAMES - len(tk))
    kept = sorted(tk + sk, key=lambda k: k[0])
    log.info("frames: %d kept (%d targeted, %d swept candidates)", len(kept), len(targeted), len(swept))
    return [(p, f"t={_mmss(t)}") for t, p, _ in kept]
