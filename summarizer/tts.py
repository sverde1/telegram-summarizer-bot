"""Summaries as voice messages: Kokoro text-to-speech in the sandbox, encoded for Telegram.

The speech itself runs in `python -m summarizer.tts_run` inside the sandbox (the text comes from the LLM and
is parsed by espeak-ng); this side prepares the text, runs it and encodes the result as an OGG Opus voice
message with ffmpeg.
"""
import json
from pathlib import Path

from . import config, proc, sandbox

RATE = 24000  # Kokoro's sample rate


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
