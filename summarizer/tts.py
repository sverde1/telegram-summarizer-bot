"""Summaries as voice messages: Kokoro text-to-speech in the sandbox, encoded for Telegram.

The speech itself runs in `python -m summarizer.tts_run` inside the sandbox (the text comes from the LLM and
is parsed by espeak-ng); this side prepares the text, runs it and encodes the result as an OGG Opus voice
message with ffmpeg.
"""
import hashlib
import json
import logging
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from . import config, fetch, proc, sandbox, stats

log = logging.getLogger(__name__)

RATE = 24000  # Kokoro's sample rate
PIECE = 400   # characters per piece: Kokoro takes at most 510 phonemes per run (about 400 characters of text)
LOAD_SECONDS = 5.0  # model load at the start of every speech run
SPEED_DEFAULT = {"cpu": 0.025, "cuda": 0.002}  # seconds per character until measured (CPU: benchmark here)

# SUMMARY_LANGUAGE -> (espeak language code, default Kokoro voice). Kokoro's voices are made for one language
# each. Japanese and Chinese are left out: espeak's phonemes read them poorly (Kokoro's own tools use others).
LANGUAGES = {"english": ("en-us", "af_heart"), "spanish": ("es", "ef_dora"), "french": ("fr-fr", "ff_siwis"),
             "italian": ("it", "if_sara"), "portuguese": ("pt-br", "pf_dora"), "hindi": ("hi", "hf_alpha")}

_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
# The model files, with the SHA-256 of the versions this bot was tested with.
MODEL_FILES = {
    "kokoro-v1.0.onnx": "7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5",
    "voices-v1.0.bin": "bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d",
}
# GitHub serves release files from a CDN host after a redirect.
DOWNLOAD_HOSTS = {"github.com", "objects.githubusercontent.com", "release-assets.githubusercontent.com"}
MAX_MODEL_BYTES = 400 * 1024 ** 2

_ready = False  # set by refresh(): models, espeak-ng and settings all fine (checked once, not per summary)


def models_dir() -> Path:
    """Folder with Kokoro's model and voices (downloaded by the bot)."""
    return config.DATA_DIR / "kokoro"


def synthesize(pieces: list[str], workdir: Path, *, voice: str, lang: str, speed: float, timeout: float,
               models: Path | None = None) -> Path:
    """Speaks the pieces in the sandbox and encodes them as a Telegram voice message.

    Args:
        pieces: The text, split at sentence ends (see split()).
        workdir: The job's directory (the sandbox's only writable place).
        voice: Kokoro voice, e.g. "af_heart".
        lang: espeak language code, e.g. "en-us".
        speed: 0.5–2.0.
        timeout: Seconds before the speech run is killed.
        models: The model folder (default models_dir()).

    Returns:
        The .ogg file (Opus, mono, 24 kbit/s).

    Raises:
        RuntimeError: The speech run or the encoding failed.
        proc.ProcCancelled: The job was cancelled.
    """
    (workdir / "pieces.json").write_text(json.dumps(pieces, ensure_ascii=False))
    pcm, ogg = workdir / "speech.pcm", workdir / "speech.ogg"
    args = ["python", "-m", "summarizer.tts_run", "/job/pieces.json", "/job/speech.pcm", voice, lang, str(speed),
            config.ESPEAK_LIB, config.ESPEAK_DATA]
    run = proc.run(sandbox.command(workdir, args, ro_binds={models or models_dir(): "/models"}), timeout=timeout)
    if run.returncode != 0 or not pcm.exists():
        raise RuntimeError(f"speech failed ({run.returncode}): {run.stderr[-500:]}")
    enc = proc.run([config.FFMPEG, "-v", "error", "-y", "-f", "f32le", "-ar", str(RATE), "-ac", "1",
                    "-i", str(pcm), "-c:a", "libopus", "-b:a", "24k", "-application", "voip", str(ogg)],
                   timeout=max(120, timeout / 4))
    pcm.unlink(missing_ok=True)
    if enc.returncode != 0 or not ogg.exists():
        raise RuntimeError(f"encoding failed: {enc.stderr[-300:]}")
    return ogg


# ---------- settings and models ----------

def language() -> tuple[str, str] | None:
    """(espeak code, voice) for SUMMARY_LANGUAGE, the voice overridden by TTS_VOICE; None if not spoken."""
    entry = LANGUAGES.get(config.SUMMARY_LANGUAGE.strip().lower())
    if not entry:
        return None
    return entry[0], config.TTS_VOICE or entry[1]


def problems() -> list[str]:
    """Why voice messages can't be made right now (empty when they can), for the admins' startup log."""
    found = []
    if not language():
        found.append(f"SUMMARY_LANGUAGE {config.SUMMARY_LANGUAGE!r} isn't one Kokoro speaks")
    if not Path(config.ESPEAK_LIB).exists() or not Path(config.ESPEAK_DATA).is_dir():
        found.append("espeak-ng isn't installed (sudo apt install espeak-ng-data), or ESPEAK_LIB/ESPEAK_DATA "
                     "point elsewhere")
    if not 0.5 <= config.TTS_SPEED <= 2.0:
        found.append(f"TTS_SPEED {config.TTS_SPEED} is outside 0.5-2.0")
    missing = [name for name in MODEL_FILES if not (models_dir() / name).exists()]
    if missing:
        found.append(f"model files missing: {', '.join(missing)}")
    elif language():
        import numpy as np
        try:
            with np.load(models_dir() / "voices-v1.0.bin") as voices:
                if language()[1] not in voices.files:
                    found.append(f"TTS_VOICE {language()[1]!r} isn't one of Kokoro's voices")
        except (OSError, ValueError) as e:
            found.append(f"the voices file can't be read: {e}")
    return found


def refresh() -> list[str]:
    """Re-checks whether voice messages can be made; returns the problems (see problems())."""
    global _ready
    found = problems()
    _ready = not found
    return found


def available() -> bool:
    """Whether the 🔊 button should be offered (as of the last refresh())."""
    return _ready


def download_models() -> None:
    """Downloads the model files that are missing, from Kokoro's GitHub release, checked against their pins.

    Raises:
        fetch.FetchError: A download failed or didn't match its checksum.
    """
    folder = models_dir()
    folder.mkdir(parents=True, exist_ok=True)
    for name, sha in MODEL_FILES.items():
        if not (folder / name).exists():
            log.info("downloading Kokoro's %s", name)
            fetch.download(_RELEASE + name, folder / name, allowed=lambda host: host in DOWNLOAD_HOSTS,
                           max_bytes=MAX_MODEL_BYTES, sha256=sha)


# ---------- what is read ----------

_URL = re.compile(r"https?://\S+")
_MARKER = re.compile(r"\[\d{1,2}:\d{2}(?::\d{2})?\]")


def clean(text: str) -> str:
    """Text as it should be read: no links, timestamps, emoji or Markdown; bullets become sentences."""
    text = _MARKER.sub("", _URL.sub("", text or ""))
    text = "".join(ch for ch in text if unicodedata.category(ch) not in ("So", "Sk", "Cs"))
    text = re.sub(r"[*_#`>|]+", "", text)
    lines = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[•\-–]|\d+[.)])\s+", "", line).strip()
        if line:
            lines.append(line if line[-1] in ".!?:;…" else line + ".")
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def video_text(summary: dict, fallback_title: str = "") -> tuple[str, str]:
    """(title, spoken text) of a video summary: title, the clickbait answer, the summary."""
    title = summary.get("title") or fallback_title
    parts = [clean(title)]
    if summary.get("is_clickbait") and summary.get("clickbait_answer"):
        parts.append("The answer: " + clean(summary["clickbait_answer"]))
    parts.append(clean(summary.get("summary", "")))
    return title, " ".join(p for p in parts if p)


def document_text(result) -> tuple[str, str]:
    """(title, spoken text) of a document summary (documents.DocResult): the book, or its chapters."""
    if result.kind == "book":
        b = result.book or {}
        title = b.get("title") or result.name
        by = f" By {clean(b['author'])}" if b.get("author") else ""
        return title, f"{clean(title)}{by} {clean(b.get('summary', ''))}".strip()
    chapters = " ".join(f"{clean(t)} {clean(s)}" for _, t, s in result.chapters)
    return result.name, chapters.strip()


def split(text: str, size: int = PIECE) -> list[str]:
    """Cuts text into pieces of at most `size` characters at sentence ends; an over-long sentence at words.

    Kokoro silently drops what follows ~510 phonemes in a piece without punctuation, so no piece may be
    longer than that, and nothing is cut away: every word is read.
    """
    pieces, current = [], ""
    for sentence in re.split(r"(?<=[.!?…])\s+", text.strip()):
        while len(sentence) > size:  # one very long sentence: cut it at a space
            cut = sentence.rfind(" ", 0, size)
            cut = cut if cut > size // 2 else size
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if current and len(current) + 1 + len(sentence) > size:
            pieces.append(current)
            current = ""
        current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return [p for p in pieces if p]


def key(text: str, lang: str, voice: str) -> str:
    """The cache key of a voice message: the text and everything that changes how it sounds."""
    return hashlib.sha256(f"{voice}\n{lang}\n{config.TTS_SPEED}\n{text}".encode()).hexdigest()


def estimate(chars: int) -> float:
    """Estimated seconds to make a voice message of this many characters."""
    return LOAD_SECONDS + chars * stats.get("tts:cpu", SPEED_DEFAULT["cpu"])


def record_speed(chars: int, seconds: float) -> None:
    """Learns the speed from a finished run (short texts are dominated by the model load)."""
    if chars >= 300:
        stats.record("tts:cpu", max(seconds - LOAD_SECONDS, 0.1) / chars)


@dataclass
class VoiceResult:
    """A voice message ready to send: a made one (ogg) or a reused one (Telegram's file id).

    The other fields mirror what the bot reads from every result (cache flag, pacing hold, no summary).
    """
    title: str
    text: str
    lang: str
    voice: str
    key: str
    file_id: str | None = None
    duration: int | None = None
    ogg: Path | None = None
    cached: bool = False
    hold: list[tuple[str, float]] | None = None
    replay_steps: list[tuple[str, float]] | None = None
    summary: dict | None = None
