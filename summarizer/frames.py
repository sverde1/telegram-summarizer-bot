"""Decide whether on-screen content matters, and if so pull a small set of distinct frames."""
import logging
import re
import subprocess
from pathlib import Path

import imagehash
from PIL import Image

from . import config

log = logging.getLogger(__name__)

SCREEN_REF = re.compile(
    r"as you can see|on (the )?screen|here are the (full )?results|this (table|chart|graph)|"
    r"look at (this|the)|the chart|shown here|link in (the )?(description|bio)|"
    r"hier (siehst|sieht)|auf dem (chart|bild)|comme vous (pouvez )?voir|como (puedes|pueden) ver",
    re.I)
NUMBER = re.compile(r"\d[\d.,]*\s?%|\$\s?\d|\d{2,}")


def decide(meta: dict, cues: list[tuple[float, str]], force: bool | None) -> tuple[bool, list[str]]:
    """Weighted signals; returns (use_frames, reasons)."""
    if force is not None:
        return force, ["forced by user" if force else "disabled by user"]
    reasons, score = [], 0
    text = " ".join(t for _, t in cues)
    minutes = max(meta.get("duration") or 0, 1) / 60
    wpm = len(text.split()) / minutes
    if wpm < 15:
        score += 3
        reasons.append(f"little or no speech ({wpm:.0f} words/min)")
    refs = len(SCREEN_REF.findall(text))
    if refs:
        score += 2 if refs < 3 else 3
        reasons.append(f"speaker refers to the screen {refs}x")
    title_nums = NUMBER.findall(meta.get("title", ""))
    if title_nums and not any(n.strip() in text for n in title_nums):
        score += 2
        reasons.append(f"title numbers {title_nums} not in transcript")
    return score >= 2, reasons


def _grab(video: Path, t: float, out: Path) -> bool:
    p = subprocess.run([config.FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{t:.2f}",
                        "-i", str(video), "-frames:v", "1", "-vf", "scale='min(1280,iw)':-2",
                        "-q:v", "2", str(out)], capture_output=True, timeout=120)
    return p.returncode == 0 and out.exists() and out.stat().st_size > 0


def _mmss(t: float) -> str:
    t = int(t)
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


def extract(video: Path, cues: list[tuple[float, str]], duration: float, workdir: Path) -> list[tuple[Path, str]]:
    """Targeted grabs just after screen references, plus an even sweep; near-duplicates removed."""
    duration = max(duration, 1)
    targeted = []
    for start, text in cues:
        if SCREEN_REF.search(text):
            targeted += [start + 2, start + 5]
    n_sweep = max(4, min(config.MAX_FRAMES, int(duration / 8)))
    sweep = [duration * (i + 0.5) / n_sweep for i in range(n_sweep)]

    # Targeted grabs first so they survive the cap, then the sweep fills remaining slots.
    ordered = [t for t in targeted if t < duration] + sweep
    fdir = workdir / "frames"
    fdir.mkdir(exist_ok=True)
    kept: list[tuple[float, Path, imagehash.ImageHash]] = []
    for t in ordered:
        if len(kept) >= config.MAX_FRAMES:
            break
        if any(abs(t - k[0]) < 1.5 for k in kept):
            continue
        out = fdir / f"f_{t:08.2f}.jpg"
        if not _grab(video, t, out):
            continue
        with Image.open(out) as im:
            h = imagehash.phash(im)
        if any(h - k[2] <= 5 for k in kept):  # talking-head segments collapse to one frame
            out.unlink()
            continue
        kept.append((t, out, h))
    kept.sort(key=lambda k: k[0])
    log.info("frames: %d kept (%d targeted candidates)", len(kept), len(targeted))
    return [(p, f"t={_mmss(t)}") for t, p, _ in kept]
