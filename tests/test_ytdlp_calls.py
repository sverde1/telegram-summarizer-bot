"""yt-dlp is asked once per job: later calls use the probe's stored info, and TikTok downloads once."""
import json
import subprocess

import pytest

from summarizer import media
from summarizer.urls import Video

YT = Video("youtube", "abcdefghijk", "https://www.youtube.com/watch?v=abcdefghijk")
TT = Video("tiktok", "123", "https://www.tiktok.com/@u/video/123")
INFO = {"id": "abcdefghijk", "title": "T", "description": "", "duration": 600, "uploader": "U",
        "subtitles": {"en": [{"ext": "vtt"}]}, "formats": [{"vcodec": "avc1"}]}
VTT = "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nhello there\n"


@pytest.fixture
def ytdlp(monkeypatch):
    """Stands in for yt-dlp: records each command and acts like it (writes the files it would)."""
    calls, failures = [], []

    def run(cmd, timeout=900):
        """Simulates one yt-dlp run."""
        calls.append(cmd)
        if failures:
            raise failures.pop(0)
        out = cmd[cmd.index("-o") + 1] if "-o" in cmd else ""
        if "-J" in cmd:
            return subprocess.CompletedProcess(cmd, 0, json.dumps(INFO), "")
        if "--write-subs" in cmd:
            open(out.replace("%(ext)s", "en.vtt"), "w").write(VTT)
        elif out:
            open(out.replace("%(ext)s", "mp4"), "wb").write(b"media")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(media, "_run", run)
    run.calls, run.failures = calls, failures
    return run


def _uses_info(cmd) -> bool:
    """Whether a yt-dlp command reads the stored info instead of looking the video up."""
    return "--load-info-json" in cmd and not any(a.startswith("https://") for a in cmd)


def test_one_lookup_then_stored_info(ytdlp, tmp_path):
    meta = media.probe(YT, tmp_path / media.INFO_JSON)
    assert json.loads((tmp_path / media.INFO_JSON).read_text())["id"] == "abcdefghijk"
    cues, lang = media.fetch_captions(YT, meta, tmp_path)
    media.download_audio(YT, tmp_path)
    media.download_video(YT, tmp_path)
    assert [c for c in ytdlp.calls if "-J" in c][0][-1] == YT.url
    later = [c for c in ytdlp.calls if "-J" not in c]
    assert len(later) == 3 and all(_uses_info(c) for c in later)  # no second lookup of the page
    assert cues and lang == "en"


def test_expired_stored_urls_are_looked_up_once_more(ytdlp, tmp_path):
    media.probe(YT, tmp_path / media.INFO_JSON)
    ytdlp.failures.append(media.MediaError("unable to download video data: HTTP Error 403: Forbidden"))
    media.download_audio(YT, tmp_path)
    audio_calls = [c for c in ytdlp.calls if "-J" not in c]
    assert _uses_info(audio_calls[0]) and audio_calls[1][-1] == YT.url and len(audio_calls) == 2


@pytest.mark.parametrize("error", [media.Blocked("HTTP Error 429: Too Many Requests"),
                                   media.MediaError("Video unavailable. This video is private")])
def test_other_errors_are_not_retried(ytdlp, tmp_path, error):
    media.probe(YT, tmp_path / media.INFO_JSON)
    ytdlp.failures.append(error)
    with pytest.raises(type(error)):
        media.download_audio(YT, tmp_path)
    assert len([c for c in ytdlp.calls if "-J" not in c]) == 1


def test_without_stored_info_the_url_is_used(ytdlp, tmp_path):
    media.download_video(YT, tmp_path)  # e.g. frames for a summary written from a cached transcript
    assert ytdlp.calls[0][-1] == YT.url and "--load-info-json" not in ytdlp.calls[0]


def test_tiktok_downloads_once_for_whisper_and_frames(ytdlp, tmp_path):
    audio = media.download_audio(TT, tmp_path)
    video = media.download_video(TT, tmp_path)
    assert audio == video and len(ytdlp.calls) == 1
    fmt = ytdlp.calls[0][ytdlp.calls[0].index("-f") + 1]
    assert fmt == "best[vcodec^=h264]/best"  # not "download": that is TikTok's watermarked file
