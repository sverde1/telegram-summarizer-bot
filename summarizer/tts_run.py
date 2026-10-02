"""Text-to-speech with Kokoro, run inside the sandbox (see tts.py).

`python -m summarizer.tts_run <pieces.json> <out.pcm> <voice> <lang> <speed> <espeak lib> <espeak data>`
reads the text pieces (split at sentence ends by tts.py), speaks them one after another with a short pause
in between, and writes raw 24 kHz mono float32 samples. The text comes from the LLM (untrusted) and goes
through espeak-ng's parser, hence the sandbox. Like docparse, this imports nothing from the bot.
"""
import json
import sys

PAUSE = 0.15  # seconds of silence between pieces (Kokoro trims each piece's own silence)
MODELS = "/models"  # where the sandbox mounts the model folder


def speak(pieces: list[str], voice: str, lang: str, speed: float, espeak_lib: str, espeak_data: str):
    """Speaks the pieces in order.

    The system's espeak-ng is passed explicitly: the copy bundled with espeakng-loader looks for its data
    in the path it was built in, which doesn't exist here.

    Returns:
        (float32 samples, sample rate).
    """
    import numpy as np
    from kokoro_onnx import EspeakConfig, Kokoro
    kokoro = Kokoro(f"{MODELS}/kokoro-v1.0.onnx", f"{MODELS}/voices-v1.0.bin",
                    espeak_config=EspeakConfig(lib_path=espeak_lib, data_path=espeak_data))
    parts, rate = [], 24000
    for piece in pieces:
        samples, rate = kokoro.create(piece, voice=voice, speed=speed, lang=lang)
        if parts:
            parts.append(np.zeros(int(PAUSE * rate), dtype=np.float32))
        parts.append(samples.astype(np.float32))
    return (np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)), rate


def main(argv: list[str]) -> int:
    """Sandbox entry point: speaks argv[0] (JSON list of pieces), writes raw float32 samples to argv[1]."""
    pieces = json.loads(open(argv[0], encoding="utf-8").read())
    samples, _ = speak(pieces, argv[2], argv[3], float(argv[4]), argv[5], argv[6])
    samples.astype("<f4").tofile(argv[1])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
