"""Pulling a small set of distinct frames at the moments the LLM asked for (plus an even sweep)."""
import logging
import re
import subprocess
from pathlib import Path

import imagehash
from PIL import Image

from . import config

log = logging.getLogger(__name__)

SCREEN_REF = re.compile(
    r"as you can see|on (the )?screen|here are the (full )?results|this (table|chart|graph|book|app|list)|"
    r"these (books|apps|tools)|look at (this|the)|the chart|shown here|right here|check (this|it) out|"
    r"link in (the )?(description|bio)|hier (siehst|sieht)|auf dem (chart|bild)|"
    r"comme vous (pouvez )?voir|como (puedes|pueden) ver",
    re.I)


def is_short(meta: dict) -> bool:
    return (meta.get("duration") or 0) <= config.SHORT_VIDEO_SEC


def regex_moments(cues: list[tuple[float, str]]) -> list[float]:
    return [start for start, text in cues if SCREEN_REF.search(text)]


def _mmss(t: float) -> str:
    t = int(t)
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


def _grab(video: Path, t: float, out: Path) -> bool:
    p = subprocess.run([config.FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{t:.2f}",
                        "-i", str(video), "-frames:v", "1", "-vf", "scale='min(1280,iw)':-2",
                        "-q:v", "2", str(out)], capture_output=True, timeout=120)
    return p.returncode == 0 and out.exists() and out.stat().st_size > 0


def _sweep(video: Path, interval: float, fdir: Path) -> list[tuple[float, Path]]:
    """One ffmpeg pass: a frame every `interval` seconds."""
    subprocess.run([config.FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", str(video),
                    "-vf", f"fps=1/{interval:.3f},scale='min(1280,iw)':-2", "-q:v", "2",
                    str(fdir / "s_%04d.jpg")], capture_output=True, timeout=900)
    return [(i * interval, p) for i, p in enumerate(sorted(fdir.glob("s_*.jpg")))]


def _spread(items: list, n: int) -> list:
    """n items evenly spread over the list (keeps first and last)."""
    if len(items) <= n:
        return items
    if n <= 1:
        return items[:n]
    return [items[round(i * (len(items) - 1) / (n - 1))] for i in range(n)]


def extract(video: Path, duration: float, moments: list[float], workdir: Path,
            sweep: bool) -> list[tuple[Path, str]]:
    """Frames just after each moment (+1 s, +3 s), plus an even sweep if `sweep`.
    Near-duplicates are dropped; at most MAX_FRAMES, preferring targeted frames."""
    duration = max(duration, 1)
    fdir = workdir / "frames"
    fdir.mkdir(exist_ok=True)

    targeted: list[tuple[float, Path]] = []
    for m in sorted(set(moments)):
        for t in (m + 1, m + 3):
            out = fdir / f"t_{t:08.2f}.jpg"
            if t < duration and _grab(video, t, out):
                targeted.append((t, out))
    swept = _sweep(video, min(max(duration / 40, 2), 20), fdir) if sweep else []

    def dedup(frames, against):
        kept = []
        for t, p in frames:
            with Image.open(p) as im:
                h = imagehash.phash(im)
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
