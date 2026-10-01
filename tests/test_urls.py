"""URL detection and classification (current behavior)."""
import pytest

from summarizer.urls import UnsupportedURL, classify, find_url


def test_find_url_takes_first_link():
    assert find_url("look https://youtu.be/abcdefghijk and https://x.y") == "https://youtu.be/abcdefghijk"
    assert find_url("no link here") is None
    assert find_url("") is None


@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=abcdefghijk",
    "https://youtube.com/watch?feature=share&v=abcdefghijk",
    "https://m.youtube.com/watch?v=abcdefghijk&pp=xyz",
    "https://youtu.be/abcdefghijk?si=tracking",
    "https://www.youtube.com/shorts/abcdefghijk",
    "https://www.youtube.com/live/abcdefghijk",
])
def test_youtube_links_normalise_to_one_id(url):
    v = classify(url)
    assert (v.platform, v.video_id) == ("youtube", "abcdefghijk")
    assert v.url == "https://www.youtube.com/watch?v=abcdefghijk"


def test_tiktok_video_and_photo_use_the_video_url():
    v = classify("https://www.tiktok.com/@some.user/video/7289499610987498794?lang=en")
    assert (v.platform, v.video_id, v.kind) == ("tiktok", "7289499610987498794", "video")
    p = classify("https://www.tiktok.com/@some.user/photo/7684449834266496258")
    assert p.kind == "photo"
    assert p.url == "https://www.tiktok.com/@some.user/video/7684449834266496258"


def test_unsupported_link_is_rejected():
    with pytest.raises(UnsupportedURL):
        classify("https://example.com/watch")
