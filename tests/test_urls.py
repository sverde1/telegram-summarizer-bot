"""Link checks: only YouTube and TikTok, no requests to anything else (SSRF), clear messages."""

import pytest

from summarizer import urls
from summarizer.urls import UnsupportedURL, check, classify, find_url


def test_find_url_takes_first_link():
    assert find_url("look https://youtu.be/abcdefghijk and https://x.y") == "https://youtu.be/abcdefghijk"
    assert find_url("no link here") is None
    assert find_url("") is None


@pytest.mark.parametrize("url", [
    "https://www.youtube.com/watch?v=abcdefghijk",
    "https://youtube.com/watch?feature=share&v=abcdefghijk",
    "https://m.youtube.com/watch?v=abcdefghijk&pp=xyz",
    "https://music.youtube.com/watch?v=abcdefghijk",
    "https://youtu.be/abcdefghijk?si=tracking",
    "http://youtu.be/abcdefghijk",
    "https://www.youtube.com/shorts/abcdefghijk",
    "https://www.youtube.com/live/abcdefghijk",
    "https://www.youtube-nocookie.com/embed/abcdefghijk",
    "https://WWW.YouTube.com./watch?v=abcdefghijk",
])
def test_youtube_links_normalise_to_one_id(url):
    v = classify(url)
    assert (v.platform, v.video_id) == ("youtube", "abcdefghijk")
    assert v.url == "https://www.youtube.com/watch?v=abcdefghijk"


def test_tiktok_video_and_photo_use_the_video_url():
    v = classify("https://www.tiktok.com/@some.user/video/7289499610987498794?lang=en")
    assert (v.platform, v.video_id, v.kind) == ("tiktok", "7289499610987498794", "video")
    p = classify("https://m.tiktok.com/@some.user/photo/7684449834266496258")
    assert p.kind == "photo"
    assert p.url == "https://www.tiktok.com/@some.user/video/7684449834266496258"


@pytest.mark.parametrize("url,message", [
    ("http://127.0.0.1:8080/admin?x=vm.tiktok.com/", urls.NOT_SUPPORTED),
    ("http://192.168.1.1/login?vm.tiktok.com/abc", urls.NOT_SUPPORTED),
    ("http://169.254.169.254/latest/meta-data?youtu.be/abcdefghijk", urls.NOT_SUPPORTED),
    ("https://evil.example/youtu.be/abcdefghijk", urls.NOT_SUPPORTED),
    ("https://youtube.com@evil.example/watch?v=abcdefghijk", urls.NOT_SUPPORTED),
    ("https://youtube.com.evil.example/watch?v=abcdefghijk", urls.NOT_SUPPORTED),
    ("http://youtu.be:8080/abcdefghijk", urls.NOT_SUPPORTED),
    ("ftp://youtu.be/abcdefghijk", urls.NOT_SUPPORTED),
    ("https://vm.tiktok.com:9999/ZM123/", urls.NOT_SUPPORTED),
    ("https://www.youtube.com/channel/UC123", urls.NOT_A_YOUTUBE_VIDEO),
    ("https://www.youtube.com/watch?v=short", urls.NOT_A_YOUTUBE_VIDEO),
    ("https://www.tiktok.com/@someone", urls.NOT_A_TIKTOK_POST),
    ("https://vm.tiktok.com/a/b/c", urls.NOT_A_TIKTOK_POST),
])
def test_bad_links_are_refused_without_any_request(url, message, monkeypatch):
    monkeypatch.setattr(urls.urllib.request, "build_opener", lambda *a: pytest.fail("made a request"))
    with pytest.raises(UnsupportedURL, match=message.split(".")[0]):
        classify(url)


@pytest.mark.parametrize("url", ["https://vm.tiktok.com/ZMabc123/", "https://vt.tiktok.com/ZSxyz/",
                                 "https://www.tiktok.com/t/ZTabc/"])
def test_short_links_pass_the_offline_check(url):
    check(url)  # no exception, no network


class _FakeOpener:
    """Plays the redirect chain a short link would go through, via the real redirect handler."""

    def __init__(self, handler, chain: list[str]):
        """Remembers the handler under test and the Location headers to hand it."""
        self.handler, self.chain = handler, chain
        self.requested = []

    def open(self, req, timeout):
        """Feeds each redirect to the handler; returns the last URL as the final page."""
        self.requested.append(req.full_url)
        current = req
        for location in self.chain:
            current = self.handler.redirect_request(current, None, 301, "Moved", {}, location)
        return _Final(current.full_url)


class _Final:
    """The final response of a resolution that ended on a page (no post found on the way)."""

    def __init__(self, url):
        """Stores the final URL."""
        self.url = url

    def geturl(self):
        """The URL the redirects ended on."""
        return self.url

    def __enter__(self):
        """Context-manager entry."""
        return self

    def __exit__(self, *exc):
        """Context-manager exit."""
        return False


def _with_chain(monkeypatch, chain):
    """Makes urls resolve short links through a fake redirect chain; returns the opener for inspection."""
    holder = {}

    def build_opener(handler):
        """Builds the fake opener around the real redirect handler."""
        holder["opener"] = _FakeOpener(handler, chain)
        return holder["opener"]

    monkeypatch.setattr(urls.urllib.request, "build_opener", build_opener)
    return holder


def test_short_link_stops_at_the_first_post_redirect(monkeypatch):
    holder = _with_chain(monkeypatch, ["https://www.tiktok.com/@u/video/123?_r=1", "https://never.example/"])
    v = classify("http://vm.tiktok.com/ZMabc/?share=1")
    assert (v.platform, v.video_id) == ("tiktok", "123")
    assert holder["opener"].requested == ["https://vm.tiktok.com/ZMabc/"]  # https, no query


@pytest.mark.parametrize("chain", [["http://www.tiktok.com/@u/video/1"],          # not https
                                   ["https://evil.example/@u/video/1"],           # leaves TikTok
                                   ["https://127.0.0.1/"],                        # internal address
                                   ["https://www.tiktok.com/foryou"]])            # ends without a post
def test_short_link_redirects_must_stay_on_tiktok(monkeypatch, chain):
    _with_chain(monkeypatch, chain)
    with pytest.raises(UnsupportedURL, match="Couldn't open that TikTok short link"):
        classify("https://vm.tiktok.com/ZMabc/")


def test_short_link_network_failure_is_a_clear_error(monkeypatch):
    class Broken:
        """An opener whose request fails like an unreachable host."""

        def open(self, req, timeout):
            """Fails the request."""
            raise urllib.error.URLError("unreachable")

    import urllib.error
    monkeypatch.setattr(urls.urllib.request, "build_opener", lambda *a: Broken())
    with pytest.raises(UnsupportedURL, match="Couldn't open that TikTok short link"):
        classify("https://vm.tiktok.com/ZMabc/")
