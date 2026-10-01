"""Frame selection and grabbing."""
from summarizer import config, frames


def test_spread_keeps_first_and_last():
    assert frames._spread(list(range(10)), 3) == [0, 4, 9]
    assert frames._spread([1, 2], 5) == [1, 2]
    assert frames._spread([1, 2, 3], 1) == [1]


def test_regex_moments_finds_screen_references():
    cues = [(1.0, "hello"), (5.0, "look at this chart"), (9.0, "as you can see here")]
    assert frames.regex_moments(cues) == [5.0, 9.0]


def test_is_short():
    assert frames.is_short({"duration": config.SHORT_VIDEO_SEC})
    assert not frames.is_short({"duration": config.SHORT_VIDEO_SEC + 1})


def test_extract_moments_and_sweep(tiny_video, tmp_path):
    got = frames.extract(tiny_video, 4.0, [0.5], tmp_path, sweep=True)
    assert 1 <= len(got) <= config.MAX_FRAMES
    assert all(p.exists() and label.startswith("t=") for p, label in got)
    seconds = [int(label.split(":")[1]) for _, label in got]  # labels look like "t=0:02"
    assert seconds == sorted(seconds)  # returned in video order


def test_identical_frames_collapse(blue_video, tmp_path):
    got = frames.extract(blue_video, 4.0, [0.2, 1.0, 2.0], tmp_path, sweep=True)
    assert len(got) == 1
