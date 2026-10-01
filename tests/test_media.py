"""Pure parts of media handling: captions, caption choice, image conversion."""
from PIL import Image

from summarizer import media

VTT = """WEBVTT
Kind: captions
Language: en

00:00:01.000 --> 00:00:03.000
<c>hello</c> there

00:00:02.500 --> 00:00:05.000
hello there
general kenobi

01:02.000 --> 01:04.000
bye
"""


def test_parse_vtt_drops_rolling_repeats_and_tags():
    assert media.parse_vtt(VTT) == [(1.0, "hello there"), (2.5, "general kenobi"), (62.0, "bye")]


def test_caption_choice_prefers_manual_then_original_language():
    base = {"language": "de", "subtitles": {}, "auto_captions": []}
    assert media._pick_caption_lang({**base, "subtitles": {"en": [], "de": []}}) == ("de", False)
    assert media._pick_caption_lang({**base, "auto_captions": ["en", "de-orig", "fr"]}) == ("de-orig", True)
    assert media._pick_caption_lang({**base, "language": "", "auto_captions": ["en"]}) == ("en", True)
    assert media._pick_caption_lang(base) is None


def test_to_jpeg_caps_the_long_edge(tmp_path):
    src = tmp_path / "big.png"
    Image.new("RGB", (4000, 1000), "red").save(src)
    out = media.to_jpeg(src, tmp_path / "out.jpg")
    with Image.open(out) as im:
        assert max(im.size) == 1568 and im.format == "JPEG"


def test_has_audio_stream(tiny_video, tmp_path):
    assert media.has_audio_stream(tiny_video)
    assert not media.has_audio_stream(tmp_path / "missing.mp4")
