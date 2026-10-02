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
