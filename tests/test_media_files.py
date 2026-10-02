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
