"""Text recognition with RapidOCR, run inside the sandbox (see ocr.py).

`python -m summarizer.rapid_ocr <out.json> <rec model or "-"> <image>...` writes, per image, the recognised
text and the mean confidence. The sandbox has no network, so models are never downloaded here: the default
models ship with the package, other languages' recognition models are installed beforehand by /ocrlang and
passed by path. Like docparse, this imports nothing from the bot (no config, no .env).
"""
import json
import sys


def recognize(images: list[str], rec_model: str | None) -> list[dict]:
    """Reads the text of each image, top to bottom.

    Args:
        images: Image paths.
        rec_model: Path of a recognition model for the language's script; None for the default (Chinese and
            English).

    Returns:
        [{"text": ..., "score": mean confidence 0–1}] per image.
    """
    from rapidocr import RapidOCR
    params = {"Global.log_level": "error",
              # One thread per process: the bot runs several OCR processes side by side.
              "EngineConfig.onnxruntime.intra_op_num_threads": 1,
              "EngineConfig.onnxruntime.inter_op_num_threads": 1}
    if rec_model:
        params["Rec.model_path"] = rec_model
    engine = RapidOCR(params=params)
    out = []
    for image in images:
        result = engine(image)
        texts, scores = list(result.txts or []), list(result.scores or [])
        out.append({"text": "\n".join(texts), "score": sum(scores) / len(scores) if scores else 0.0})
    return out


def main(argv: list[str]) -> int:
    """Sandbox entry point: recognises argv[2:], writes the JSON list to argv[0]."""
    out, rec, images = argv[0], argv[1], argv[2:]
    with open(out, "w") as f:
        json.dump(recognize(images, None if rec == "-" else rec), f, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
