"""Live streams and videos of unknown length are refused with an explanation."""
import json
import subprocess

import pytest

from summarizer import media, pipeline
from summarizer.urls import classify

from helpers import meta

LINK = "https://youtu.be/abcdefghijk"


def _probe_with(monkeypatch, **fields):
    """Makes yt-dlp's metadata answer contain the given fields."""
    info = {"id": "abcdefghijk", "title": "T", "duration": 600, "formats": [{"vcodec": "h264"}], **fields}
    monkeypatch.setattr(media, "_ytdlp", lambda *a, **k: subprocess.CompletedProcess([], 0, json.dumps(info), ""))


@pytest.mark.parametrize("fields,text", [({"live_status": "is_live", "duration": None}, "Live streams aren't supported"),
                                         ({"is_live": True}, "Live streams aren't supported"),
                                         ({"live_status": "post_live"}, "Live streams aren't supported"),
                                         ({"live_status": "is_upcoming"}, "hasn't started yet")])
def test_live_and_upcoming_streams_are_refused(monkeypatch, fields, text):
    _probe_with(monkeypatch, **fields)
    with pytest.raises(pipeline.PipelineError) as e:
        pipeline.run(LINK, lambda *a: None)
    assert text in str(e.value) and not str(e.value).startswith("Couldn't load")


def test_finished_stream_with_recording_is_fine(monkeypatch):
    _probe_with(monkeypatch, live_status="was_live")
    assert media.probe(classify(LINK))["duration"] == 600


def test_unknown_length_is_refused(monkeypatch, llm):
    monkeypatch.setattr(media, "probe", lambda v: meta(duration=0))
    with pytest.raises(pipeline.PipelineError, match="Couldn't determine this video's length"):
        pipeline.run(LINK, lambda *a: None)


def test_carousel_without_music_has_no_length_and_is_fine(monkeypatch, llm, tmp_path):
    monkeypatch.setattr(media, "probe", lambda v: meta(duration=0, is_carousel=True))
    slide = tmp_path / "01.jpg"
    from PIL import Image
    Image.new("RGB", (10, 10)).save(slide)
    monkeypatch.setattr(media, "download_carousel", lambda v, w: [slide])
    monkeypatch.setattr(media, "download_thumbnail", lambda m, w: None)
    assert pipeline.run("https://www.tiktok.com/@u/photo/123", lambda *a: None).summary


@pytest.mark.parametrize("error,text", [("[youtube] x: This live stream recording is not available.", "Live streams"),
                                        ("[youtube] x: This live event will begin in 3 hours.", "hasn't started"),
                                        ("[youtube] x: Premieres in 20 minutes", "hasn't started")])
def test_yt_dlp_live_errors_get_the_same_message(monkeypatch, error, text):
    def fail(*a, **k):
        """yt-dlp refusing the stream."""
        raise media.MediaError(error)

    monkeypatch.setattr(media, "_ytdlp", fail)
    with pytest.raises(media.NotProcessable, match=text):
        media.probe(classify(LINK))


def test_other_download_errors_pass_through(monkeypatch):
    monkeypatch.setattr(media, "_ytdlp", lambda *a, **k: (_ for _ in ()).throw(media.MediaError("Video unavailable")))
    with pytest.raises(media.MediaError, match="Video unavailable") as e:
        media.probe(classify(LINK))
    assert not isinstance(e.value, media.NotProcessable)
