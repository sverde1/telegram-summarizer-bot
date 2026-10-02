"""How much memory a Whisper transcription needs, and whether the machine has room for it right now.

Whisper holds the whole decoded audio in memory several times over (the samples, a speech-only copy made by
the voice-activity filter, and the log-mel features), so a 3-hour video needs a few GB. Before transcribing,
the bot checks that this fits within WHISPER_RAM_FRACTION of the machine's RAM and in what's free right now;
if not, the job waits (see bot) instead of pushing the machine into swap or the OOM killer.
"""
from pathlib import Path

from . import config

GB = 1024 ** 3
# Memory for the decoded audio and what Whisper derives from it: 16 000 float32 samples per second, about
# three copies in flight (samples, VAD speech copy, log-mel features).
_BYTES_PER_AUDIO_SECOND = 3 * 16000 * 4
# Memory for the model itself when it isn't loaded yet (CPU int8 and GPU float16 both stay below these).
_MODEL_BYTES = {"tiny": 0.2 * GB, "base": 0.3 * GB, "small": 0.6 * GB, "medium": 1.6 * GB}
_DEFAULT_MODEL_BYTES = 3.5 * GB  # large-v2 / large-v3 and anything unknown


class NeedsMemory(Exception):
    """There isn't enough free memory to transcribe this video right now; the job should wait."""

    def __init__(self, needed: int, duration: float = 0):
        """Stores how many bytes the transcription needs, and the audio length it was estimated for (so a
        waiting job can re-estimate later, e.g. once the model is loaded and no longer needs counting)."""
        super().__init__(f"needs {needed / GB:.1f} GB")
        self.needed = needed
        self.duration = duration


def _meminfo(field: str) -> int:
    """One value from /proc/meminfo, in bytes (0 if unavailable)."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith(field + ":"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def total() -> int:
    """The machine's RAM, in bytes."""
    return _meminfo("MemTotal")


def available() -> int:
    """Memory available for new work right now (MemAvailable, which counts reclaimable cache), in bytes."""
    return _meminfo("MemAvailable")


def used_by_bot() -> int:
    """This process's resident memory (RSS), in bytes."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def whisper_needs(duration: float, model_loaded: bool) -> int:
    """Estimated memory to transcribe `duration` seconds of audio, in bytes.

    Args:
        duration: Audio length in seconds.
        model_loaded: Whether the Whisper model is already in memory (then it's part of the bot's RSS
            already and isn't counted again).
    """
    model = 0 if model_loaded else _MODEL_BYTES.get(config.WHISPER_MODEL, _DEFAULT_MODEL_BYTES)
    return int(duration * _BYTES_PER_AUDIO_SECOND + model)


def cap() -> int:
    """The most memory the bot may use for a transcription: WHISPER_RAM_FRACTION of RAM, in bytes."""
    return int(total() * config.WHISPER_RAM_FRACTION)


def can_ever_fit(needed: int) -> bool:
    """Whether a transcription needing this much could fit at all, even with nothing else running."""
    return needed <= cap()


def fits_now(needed: int) -> bool:
    """Whether a transcription needing this much fits right now: within the cap and in free memory."""
    return used_by_bot() + needed <= cap() and needed <= available()
