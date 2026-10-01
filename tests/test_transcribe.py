"""Whisper helpers that need no model: decoding and the time estimate."""
from summarizer import config, stats, transcribe


def test_decode_gives_16khz_mono_float32(tiny_video):
    samples = transcribe._decode(str(tiny_video))
    assert samples.dtype.name == "float32"
    assert abs(len(samples) - 4 * 16000) < 1600  # 4 s of audio, give or take container padding


def test_estimate_uses_measured_speed_and_model_load(monkeypatch):
    monkeypatch.setattr(transcribe, "_model", None)
    stats.record(transcribe.speed_key(), 0.25)
    assert transcribe.estimate(100) == 100 * 0.25 + 15  # +15 s while the model isn't loaded
    monkeypatch.setattr(transcribe, "_model", object())
    assert transcribe.estimate(100) == 25


def test_speed_key_names_device_and_model():
    assert transcribe.speed_key() == f"whisper:{config.WHISPER_DEVICE}:{config.WHISPER_MODEL}"
