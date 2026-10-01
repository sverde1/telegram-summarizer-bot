"""Speech-to-text with faster-whisper. Device/model come from config (CPU now, GPU later)."""
import logging
import threading
import time

import numpy as np

from . import config, proc, stats

log = logging.getLogger(__name__)

_model = None
_lock = threading.Lock()  # one transcription at a time; the model stays loaded between jobs


def _load(device: str, compute_type: str):
    """Loads the Whisper model and proves it works with a real inference.

    Args:
        device: "cpu" or "cuda".
        compute_type: ctranslate2 compute type, e.g. "int8" or "float16".

    Returns:
        The loaded `faster_whisper.WhisperModel`.

    Raises:
        Exception: Loading or the test inference failed (e.g. CUDA libraries missing).
    """
    from faster_whisper import WhisperModel  # imported here: slow import, only needed once a job transcribes
    m = WhisperModel(config.WHISPER_MODEL, device=device, compute_type=compute_type,
                     cpu_threads=config.WHISPER_CPU_THREADS)
    # Building the model succeeds even when CUDA libs are missing; the failure only shows at the
    # first encode. So run a real 1-second inference now (16000 samples = 1 s at 16 kHz).
    segs, _ = m.transcribe(np.zeros(16000, dtype=np.float32), language=None)
    list(segs)  # segments are a lazy generator: consuming it is what actually runs the model
    return m


def is_loaded() -> bool:
    """Whether the Whisper model is already in memory (loaded by an earlier transcription)."""
    return _model is not None


def get_model():
    """Returns the Whisper model, loading it on first use.

    It stays loaded for the life of the process: loading takes seconds, longer than transcribing a short
    clip. If the configured GPU device fails, falls back to CPU int8 so transcription keeps working.

    Returns:
        The loaded `faster_whisper.WhisperModel`.

    Raises:
        Exception: The model couldn't be loaded on the CPU either.
    """
    global _model
    if _model is None:
        device, ctype = config.WHISPER_DEVICE, config.WHISPER_COMPUTE_TYPE
        try:
            _model = _load(device, ctype)
        except Exception as e:
            if device == "cpu":
                raise
            log.error("whisper on %s failed (%s); falling back to CPU int8", device, e)
            device, ctype = "cpu", "int8"
            _model = _load(device, ctype)
        log.info("whisper model %s loaded on %s (%s)", config.WHISPER_MODEL, device, ctype)
    return _model


def _decode(path: str) -> np.ndarray:
    """Decodes an audio file to the 16 kHz mono float32 samples Whisper expects.

    Uses ffmpeg rather than faster-whisper's own decoder: that one passes `metadata_errors` to
    `av.open()`, which the PyAV version installed here no longer accepts.

    Args:
        path: Any audio or video file ffmpeg can read.

    Returns:
        The samples as a 1-D float32 array.

    Raises:
        RuntimeError: ffmpeg couldn't decode the file.
    """
    try:
        p = proc.run([config.FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", path,
                      "-ac", "1", "-ar", "16000", "-f", "f32le", "-"], timeout=1800, text=False)
    except proc.ProcTimeout:
        raise RuntimeError("decoding the audio timed out")
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg could not decode audio: {p.stderr.decode(errors='replace')[-300:]}")
    return np.frombuffer(p.stdout, dtype=np.float32)


def transcribe(audio_path: str) -> tuple[list[tuple[float, str]], str, float]:
    """Transcribes an audio file, detecting its language.

    Language is always auto-detected: forcing it (or using an English-only *.en model) on
    non-English audio makes whisper produce fluent, invented English instead of an error.
    Also records the measured speed for ETA estimates.

    Args:
        audio_path: The audio file.

    Returns:
        ([(start_seconds, text)], language code, language probability). The list is empty when no
        speech was found (e.g. music only).

    Raises:
        RuntimeError: The audio couldn't be decoded.
    """
    with _lock:
        model = get_model()
        t0 = time.monotonic()
        audio = _decode(audio_path)
        # The voice-activity filter skips silence and music, where Whisper tends to hallucinate text.
        segments, info = model.transcribe(audio, language=None, vad_filter=True)
        cues = []
        for s in segments:  # the generator transcribes as it's consumed: a cancel stops it between segments
            proc.check_cancelled()
            if s.text.strip():
                cues.append((s.start, s.text.strip()))
        audio_sec = len(audio) / 16000
        # Very short clips are dominated by fixed overhead and would skew the realtime factor.
        if audio_sec > 5:
            stats.record(speed_key(), (time.monotonic() - t0) / audio_sec)
    return cues, info.language, info.language_probability


def speed_key() -> str:
    """Returns the stats key for the measured speed of the configured device and model."""
    return f"whisper:{config.WHISPER_DEVICE}:{config.WHISPER_MODEL}"


def estimate(audio_sec: float) -> float:
    """Estimates how long transcribing will take.

    Args:
        audio_sec: Length of the audio in seconds.

    Returns:
        Seconds, from the measured realtime factor (+ model load if not loaded yet).
    """
    # Defaults until a real run is measured: rough realtime factors for `small` on CPU int8 vs a GPU.
    # 15 s: a rough allowance for loading the model and its self-test, paid once per process.
    factor = stats.get(speed_key(), 0.5 if config.WHISPER_DEVICE == "cpu" else 0.05)
    return audio_sec * factor + (0 if _model else 15)
