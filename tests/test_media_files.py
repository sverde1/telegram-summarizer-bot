"""Files users send (voice messages, audio, video): read only inside the sandbox, streams told apart."""
import subprocess
from pathlib import Path

import pytest

from summarizer import config, frames, media, proc, transcribe

FF = [config.FFMPEG, "-v", "error", "-y"]


def _make(path: Path, *args: str) -> Path:
    """Renders a small media file with the bot's ffmpeg."""
    subprocess.run([*FF, *args, str(path)], check=True, timeout=60)
    return path


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    """Generated test media: a video with sound, a silent video, an mp3, an mp3 with cover art, a voice note."""
    d = tmp_path_factory.mktemp("media")
    tone = ["-f", "lavfi", "-i", "sine=frequency=440:duration=3"]
    picture = ["-f", "lavfi", "-i", "testsrc=size=320x240:rate=10:duration=3"]
    out = {
        "video": _make(d / "video.mp4", *picture, *tone, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                       "-shortest"),
        "silent_video": _make(d / "silent.mp4", *picture, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an"),
        "mp3": _make(d / "song.mp3", *tone, "-c:a", "libmp3lame"),
        "voice": _make(d / "voice.ogg", *tone, "-c:a", "libopus"),
    }
    cover = _make(d / "cover.png", "-f", "lavfi", "-i", "color=c=red:size=64x64", "-frames:v", "1")
    out["cover_mp3"] = _make(d / "cover.mp3", *tone, "-i", str(cover), "-map", "0:a", "-map", "1:v",
                             "-c:a", "libmp3lame", "-c:v", "png", "-disposition:v", "attached_pic")
    (d / "broken.mp4").write_bytes(b"%PDF-1.4 not a video" * 20)
    out["broken"] = d / "broken.mp4"
    return out


@pytest.fixture
def job(tmp_path, files):
    """A job directory, and a helper that copies one of the test files into it."""
    def put(name):
        """The test file `name`, copied into the job directory."""
        dst = tmp_path / files[name].name
        dst.write_bytes(files[name].read_bytes())
        return dst
    put.dir = tmp_path
    return put


@pytest.fixture
def argv(monkeypatch):
    """Records the commands run through proc.run."""
    seen = []
    real = proc.run
    monkeypatch.setattr(proc, "run", lambda cmd, **kw: seen.append(cmd) or real(cmd, **kw))
    return seen


@pytest.mark.parametrize("name,audio,video", [
    ("video", True, True), ("silent_video", False, True), ("mp3", True, False), ("voice", True, False),
    ("cover_mp3", True, False),  # cover art isn't video
])
def test_streams_are_told_apart(job, argv, name, audio, video):
    info = media.probe_file(job(name), job.dir)
    assert (info["has_audio"], info["has_video"]) == (audio, video) and 2.5 < info["duration"] < 3.5
    assert all(cmd[0] == "bwrap" for cmd in argv)  # read only in the sandbox


def test_probe_without_ffprobe(job, monkeypatch):
    monkeypatch.setattr(config, "FFPROBE", None)
    assert media.probe_file(job("cover_mp3"), job.dir)["has_video"] is False
    info = media.probe_file(job("video"), job.dir)
    assert info["has_audio"] and info["has_video"] and 2.5 < info["duration"] < 3.5


def test_unreadable_file(job):
    with pytest.raises(media.MediaError):
        media.probe_file(job("broken"), job.dir)


def test_decoding_a_users_file_happens_in_the_sandbox(job, argv):
    samples = transcribe._decode(str(job("voice")), job.dir)
    assert 2.5 * 16000 < len(samples) < 3.5 * 16000 and argv[0][0] == "bwrap"
    assert not (job.dir / "decoded.f32").exists()


def test_frames_from_a_users_file_in_the_sandbox(job, argv):
    images = frames.extract(job("video"), 3, [0.5], job.dir, sweep=True, sandboxed=True)
    assert images and all(cmd[0] == "bwrap" for cmd in argv)


def test_platform_downloads_keep_running_ffmpeg_directly(job, argv):
    frames._grab(job("video"), 1, job.dir / "f.jpg")
    assert argv[0][0] == config.FFMPEG


# ---------- through the pipeline ----------

import shutil  # noqa: E402

from summarizer import db, memory, pipeline, summarize  # noqa: E402

from helpers import SUMMARY, FakeConversation  # noqa: E402

SHA = "a" * 64


@pytest.fixture
def ai(monkeypatch):
    """A fake LLM answering with a plain summary (no frames requested); returns the conversations."""
    convs = []

    def factory(backend=None, model=None):
        """One fake conversation."""
        c = FakeConversation([{**SUMMARY, "title": "Battery talk", "needs_frames": False, "frame_moments": []}])
        convs.append(c)
        return c

    monkeypatch.setattr(summarize, "conversation", factory)
    return convs


@pytest.fixture
def whisper(monkeypatch):
    """A fake Whisper: returns `whisper.cues` and records where it was asked to decode."""
    calls = []

    def fake(path, sandbox_dir=None):
        """Records the call; returns the prepared cues."""
        calls.append((path, sandbox_dir))
        return list(fake.cues), "en", 0.9

    fake.cues = [(0.0, "hello there"), (1.0, "this is a test recording about batteries " * 5)]
    fake.calls = calls
    monkeypatch.setattr(pipeline.transcribe, "transcribe", fake)
    monkeypatch.setattr(memory, "fits_now", lambda needed: True)
    return fake


def _run_file(job, name, label="🎬 holiday.mp4", **kw):
    """Puts a test file in the job directory as the bot would (vid.* / audio.*) and runs it."""
    src = job(name)
    info = media.probe_file(src, job.dir)
    dst = job.dir / (("vid" if info["has_video"] else "audio") + src.suffix)
    src.rename(dst)
    return pipeline.run_file(dst, label, info, kw.pop("sha", SHA), lambda *a: None, workdir=job.dir, **kw)


def test_a_video_file_is_summarized_with_frames_from_the_file(job, ai, whisper):
    whisper.cues = []  # no speech: the picture is the content
    r = _run_file(job, "video")
    assert r.summary["title"] == "Battery talk" and r.frames_used and r.platform == "file"
    assert ai[0].sent[0][0]  # frames were attached on the first turn
    assert whisper.calls and whisper.calls[0][1] == job.dir  # decoded in the sandbox
    row = db.get_video("file", SHA)
    assert row["url"] == "" and row["meta"]["title"] == "Video file"  # no file name in the shared cache
    assert job.dir.exists()  # the bot owns the directory


def test_an_audio_file_gets_no_frames(job, ai, whisper):
    r = _run_file(job, "mp3", "🎵 song.mp3")
    assert r.summary and not r.frames_used and not ai[0].sent[0][0]


def test_cover_art_is_not_a_picture_to_look_at(job, ai, whisper):
    r = _run_file(job, "cover_mp3", "🎵 cover.mp3")
    assert not r.frames_used


def test_audio_without_speech(job, ai, whisper):
    whisper.cues = []
    with pytest.raises(pipeline.PipelineError, match="No speech found"):
        _run_file(job, "voice", "🎤 Voice message")
    assert ai == []


def test_a_repeat_by_someone_else_reuses_it_without_the_first_name(job, ai, whisper, tmp_path_factory, files):
    _run_file(job, "voice", "🎤 Voice message")
    other = tmp_path_factory.mktemp("other")
    shutil.copy(files["voice"], other / "audio.ogg")
    info = media.probe_file(other / "audio.ogg", other)
    rid = db.add_request(61, "x", "summary")
    r = pipeline.run_file(other / "audio.ogg", "🎵 my-secret-name.ogg", info, SHA, lambda *a: None,
                          workdir=other, request_id=rid, hide_cache_from=61)
    assert r.cached and r.replay_steps and len(ai) == 1  # paced like other cached answers
    assert r.url == "🎵 my-secret-name.ogg" and "my-secret" not in str(db.get_video("file", SHA))


def test_waiting_for_memory_keeps_the_file(job, ai, whisper, monkeypatch):
    monkeypatch.setattr(memory, "fits_now", lambda needed: False)
    with pytest.raises(memory.NeedsMemory):
        _run_file(job, "voice", "🎤 Voice message")
    assert list(job.dir.glob("audio.*"))  # still there for the retry: nothing to download again


def test_a_file_summary_has_no_clickbait_section_and_names_the_file():
    import bot
    r = pipeline.Result("file", SHA, "🎤 Voice message", {"title": "Audio file", "duration": 42}, "t", "whisper-small",
                        "en", {**SUMMARY, "title": "Battery talk", "_stats": {}})
    text = bot.render(r)[0]
    assert "Clickbait" not in text and "🎤 Voice message" in text and "Battery talk" in text
    assert bot._transcript_name(r) == "Voice message transcript.txt"
