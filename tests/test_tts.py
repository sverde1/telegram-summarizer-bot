"""Text-to-speech: Kokoro in the sandbox (real run when the models are available), text preparation."""
import os
import subprocess
from pathlib import Path

import pytest

from summarizer import config, tts

MODELS = os.environ.get("KOKORO_MODELS")


@pytest.mark.slow
@pytest.mark.skipif(not MODELS, reason="set KOKORO_MODELS to a folder with kokoro-v1.0.onnx and voices-v1.0.bin")
def test_kokoro_speaks_in_the_sandbox(tmp_path):
    ogg = tts.synthesize(["Hello there. This is a test of the voice message.", "And a second piece."], tmp_path,
                         voice="af_heart", lang="en-us", speed=1.0, timeout=300, models=Path(MODELS))
    probe = subprocess.run([config.FFMPEG, "-i", str(ogg)], capture_output=True, text=True).stderr
    assert "opus" in probe and ogg.stat().st_size > 2000
    seconds = [float(x) for x in __import__("re").findall(r"Duration: 00:00:(\d+\.\d+)", probe)]
    assert seconds and seconds[0] > 2


# ---------- what is read ----------

def test_clean_reads_naturally():
    text = "• Point one\n- Two, see https://x.y/z [1:23] 🎉\n**Bold** heading\n1. Numbered"
    assert tts.clean(text) == "Point one. Two, see. Bold heading. Numbered."


def test_split_keeps_every_word_and_respects_the_size():
    text = "Short one. " + "word " * 300 + "end of a long sentence. Last."
    pieces = tts.split(text, 100)
    assert all(len(p) <= 100 for p in pieces)
    assert " ".join(pieces).split() == text.split()  # nothing dropped, nothing invented


def test_video_and_document_text():
    title, text = tts.video_text({"title": "T", "is_clickbait": True, "clickbait_answer": "Yes, it works.",
                                  "summary": "• Fact one\n• Fact two"})
    assert (title, text) == ("T", "T. The answer: Yes, it works. Fact one. Fact two.")
    from summarizer import documents
    book = documents.DocResult("book", "b.pdf", {}, book={"title": "Pets", "author": "Ana", "summary": "Cats."})
    assert tts.document_text(book) == ("Pets", "Pets. By Ana. Cats.")
    each = documents.DocResult("each", "b.pdf", {}, chapters=[(0, "Cats", "They sleep"), (1, "Dogs", "Loyal")])
    assert tts.document_text(each) == ("b.pdf", "Cats. They sleep. Dogs. Loyal.")


def test_key_changes_with_anything_that_changes_the_sound(monkeypatch):
    base = tts.key("Hello.", "en-us", "af_heart")
    assert base != tts.key("Hello.", "en-us", "am_adam") and base != tts.key("Hello!", "en-us", "af_heart")
    monkeypatch.setattr(config, "TTS_SPEED", 1.2)
    assert base != tts.key("Hello.", "en-us", "af_heart")


# ---------- settings and models ----------

def _voices(folder, names=("af_heart",)):
    """Writes a stand-in model folder: a dummy model and a voices file with these voices."""
    import numpy as np
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "kokoro-v1.0.onnx").write_bytes(b"x")
    with open(folder / "voices-v1.0.bin", "wb") as f:
        np.savez(f, **{n: np.zeros((1, 256), dtype=np.float32) for n in names})


def test_ready_when_everything_is_there(monkeypatch, tmp_path):
    monkeypatch.setattr(tts, "models_dir", lambda: tmp_path)
    _voices(tmp_path)
    assert tts.refresh() == [] and tts.available() and tts.language() == ("en-us", "af_heart")


@pytest.mark.parametrize("setting,value,problem", [
    ("SUMMARY_LANGUAGE", "Slovenian", "isn't one Kokoro speaks"),
    ("ESPEAK_LIB", "/nonexistent.so", "espeak-ng isn't installed"),
    ("TTS_SPEED", 3.0, "outside 0.5-2.0"),
    ("TTS_VOICE", "zz_nobody", "isn't one of Kokoro's voices"),
])
def test_problems_turn_voice_messages_off(monkeypatch, tmp_path, setting, value, problem):
    monkeypatch.setattr(tts, "models_dir", lambda: tmp_path)
    _voices(tmp_path)
    monkeypatch.setattr(config, setting, value)
    found = tts.refresh()
    assert any(problem in p for p in found) and not tts.available()


def test_missing_models_are_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(tts, "models_dir", lambda: tmp_path)
    assert any("model files missing" in p for p in tts.refresh())


def test_models_come_only_from_the_release_with_their_pins(monkeypatch, tmp_path):
    from summarizer import fetch
    monkeypatch.setattr(tts, "models_dir", lambda: tmp_path)
    seen = []
    monkeypatch.setattr(fetch, "download", lambda url, dest, **kw: seen.append((url, kw)) or dest.write_bytes(b"x"))
    tts.download_models()
    assert [u for u, _ in seen] == [tts._RELEASE + n for n in tts.MODEL_FILES]
    assert all(kw["sha256"] == tts.MODEL_FILES[u.rsplit("/", 1)[1]] for u, kw in seen)
    allowed = seen[0][1]["allowed"]
    assert allowed("release-assets.githubusercontent.com") and not allowed("evil.com")
    tts.download_models()
    assert len(seen) == 2  # present files aren't downloaded again


# ---------- stored data and limits ----------

def test_voice_requests_are_not_counted_in_the_daily_limit():
    from summarizer import db
    db.add_request(60, "u", "summary")
    v = db.add_request(60, "u", "voice")
    db.update_request(v, status="done")
    assert db.daily_usage(60)[0] == 1 and db.tts_usage(60)[0] == 1
    db.update_request(v, cached=1)  # a reused voice message is free
    assert db.tts_usage(60)[0] == 0


def test_voice_messages_expire(monkeypatch):
    from summarizer import db
    db.save_voice("k", "FILE", 12)
    assert db.get_voice("k", 7 * 86400)["file_id"] == "FILE"
    assert db.get_voice("k", -1) is None and db.get_voice("k", 7 * 86400) is None  # expired and deleted


async def test_limit_voice_command(app, telegram):
    import access
    from conftest import ADMIN_ID, msg_update, send
    access.set_state(60, "allowed")
    await send(app, msg_update(ADMIN_ID, "/limit voice 3"))
    assert telegram.texts()[-1] == "✅ Voice-message limit for everyone: 3."
    await send(app, msg_update(60, "/limit"))
    assert telegram.texts()[-1].split("\n")[2] == "🔊 Today: 0 of your 3 new voice messages (last 24 h). 3 left."



# ---------- device ----------

def test_cuda_when_available_else_cpu(monkeypatch, tmp_path):
    import sys
    import types
    from summarizer import proc
    monkeypatch.setattr(config, "TTS_DEVICE", "cuda")
    monkeypatch.setitem(sys.modules, "onnxruntime", types.SimpleNamespace(get_available_providers=lambda: ["CUDAExecutionProvider"]))
    assert tts.device() == "cuda" and tts.estimate(1000) < 10
    seen = []

    def run(cmd, **kw):
        """Records the sandbox command; pretends Kokoro and ffmpeg worked."""
        seen.append(cmd)
        out = tmp_path / ("speech.pcm" if len(seen) == 1 else "speech.ogg")
        out.write_bytes(b"x")
        return types.SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(proc, "run", run)
    tts.synthesize(["Hi."], tmp_path, voice="af_heart", lang="en-us", speed=1.0, timeout=60, models=tmp_path)
    assert "ONNX_PROVIDER" in seen[0] and "CUDAExecutionProvider" in seen[0] and "prlimit" not in seen[0]
    monkeypatch.setattr(tts, "_cuda_ok", None)
    monkeypatch.setitem(sys.modules, "onnxruntime", types.SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"]))
    assert tts.device() == "cpu"
