"""Whisper helpers that need no model: decoding, the time estimate and progress reports."""
import numpy as np

from summarizer import config, pipeline, stats, transcribe


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


class _Segment:
    """A faster-whisper segment."""

    def __init__(self, start, end, text):
        """Its times and text."""
        self.start, self.end, self.text = start, end, text


def test_progress_is_reported_per_segment(monkeypatch):
    class Model:
        """Yields three segments of a 100 s file."""

        def transcribe(self, audio, **kw):
            """The segments and the language info."""
            info = type("Info", (), {"language": "en", "language_probability": 0.9})
            return iter([_Segment(0, 30, "a"), _Segment(30, 60, " "), _Segment(60, 120, "c")]), info

    monkeypatch.setattr(transcribe, "_model", Model())
    monkeypatch.setattr(transcribe, "_decode", lambda path, sandbox_dir=None: np.zeros(100 * 16000, np.float32))
    seen = []
    cues, lang, _ = transcribe.transcribe("a.wav", on_progress=lambda done, total: seen.append((done, total)))
    assert seen == [(30, 100), (60, 100), (100, 100)] and [c[1] for c in cues] == ["a", "c"] and lang == "en"


def test_status_shows_the_percentage_and_the_live_estimate(monkeypatch):
    shown = []
    st = pipeline.Status(lambda text, eta: shown.append((text, eta)))
    now = [1000.0]
    monkeypatch.setattr(pipeline.time, "monotonic", lambda: now[0])
    report = pipeline.whisper_progress(st, "🗣 transcribing…", estimate=500, rest=60)
    now[0] += 5
    report(1, 1000)  # under the update interval: not shown
    now[0] += 10
    report(10, 1000)  # 1 %: the first estimate counts down
    now[0] += 10
    report(100, 1000)  # 10 %: this run's rate (25 s for 100 s of audio) for the rest, plus the later steps
    now[0] += 1
    report(500, 1000)  # too soon after the last update
    now[0] += 30
    report(1000, 1000)  # done, but shown as 99 % until Whisper returns
    assert [t for t, _ in shown] == ["🗣 transcribing… 1 %", "🗣 transcribing… 10 %", "🗣 transcribing… 99 %"]
    assert shown[0][1] == 485 and shown[1][1] == 25 / 100 * 900 + 60
