"""Speech-to-text with faster-whisper. Device/model come from config (CPU now, GPU later)."""
import logging
import subprocess
import threading

import numpy as np

from . import config

log = logging.getLogger(__name__)

_model = None
_lock = threading.Lock()  # one transcription at a time; the model stays loaded between jobs


def _load(device: str, compute_type: str):
    from faster_whisper import WhisperModel
    m = WhisperModel(config.WHISPER_MODEL, device=device, compute_type=compute_type,
                     cpu_threads=config.WHISPER_CPU_THREADS)
    # Building the model succeeds even when CUDA libs are missing; the failure only shows at the
    # first encode. So run a real 1-second inference now.
    segs, _ = m.transcribe(np.zeros(16000, dtype=np.float32), language=None)
    list(segs)
    return m


def get_model():
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
    """16 kHz mono float32 via ffmpeg (avoids faster-whisper's PyAV decoder, which breaks on new PyAV)."""
    p = subprocess.run([config.FFMPEG, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", path,
                        "-ac", "1", "-ar", "16000", "-f", "f32le", "-"], capture_output=True, timeout=1800)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg could not decode audio: {p.stderr.decode(errors='replace')[-300:]}")
    return np.frombuffer(p.stdout, dtype=np.float32)


def transcribe(audio_path: str) -> tuple[list[tuple[float, str]], str, float]:
    """Returns ([(start, text)], language, language_probability).

    Language is always auto-detected: forcing it (or using an English-only *.en model) on
    non-English audio makes whisper produce fluent, invented English instead of an error.
    """
    with _lock:
        segments, info = get_model().transcribe(_decode(audio_path), language=None, vad_filter=True)
        cues = [(s.start, s.text.strip()) for s in segments if s.text.strip()]
    return cues, info.language, info.language_probability
