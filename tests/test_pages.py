"""London Real (and future video pages): checking the link, finding the video it plays, the footer's links."""
from pathlib import Path

import pytest

from summarizer import db, fetch, pages, pipeline, urls
from tgbot import render

PAGE = "https://londonreal.tv/grant-cardone-goes-all-in-on-bitcoin-why-mass-adoption-is-inevitable/"
JSON_LD = ('<script type="application/ld+json">{"@type":"VideoObject",'
           '"embedUrl":"https:\\/\\/www.youtube.com\\/embed\\/TeRAr7tE48A"}</script>')
PROMO = '<iframe src="https://www.youtube.com/embed/l0GzDCZs9Yc?iv_load_policy=3&amp;rel=0"></iframe>'
PLAYER = '<iframe src="https://www.youtube.com/embed/TeRAr7tE48A?origin=https://londonreal.tv&amp;modestbranding=1">'


@pytest.mark.parametrize("url", [PAGE, PAGE.replace("londonreal.tv", "www.londonreal.tv"),
                                 PAGE.replace("londonreal.tv", "londonreal.tv."), PAGE + "?utm_source=x"])
def test_episode_links_pass_without_network(url):
    urls.check(url)  # the conftest blocks sockets: no request is made


@pytest.mark.parametrize("url, message", [
    ("https://londonreal.tv/", urls.NOT_AN_EPISODE), ("https://londonreal.tv/episodes/x/y/", urls.NOT_AN_EPISODE),
    ("https://londonreal.tv/?p=1", urls.NOT_AN_EPISODE), ("https://academy.londonreal.tv/x/", urls.NOT_SUPPORTED),
    ("https://londonreal.tv.evil.com/x/", urls.NOT_SUPPORTED)])
def test_other_links_are_refused(url, message):
    with pytest.raises(urls.UnsupportedURL, match=message.split(".")[0]):
        urls.check(url)


@pytest.mark.parametrize("html, vid", [
    (PROMO + JSON_LD + PLAYER, "TeRAr7tE48A"),  # the page's own structured data wins
    (PROMO + PLAYER, "TeRAr7tE48A"),  # else the main player (origin=the site), even after a promo
    (PROMO, "l0GzDCZs9Yc"),  # else the first embedded video
    ("<p>members only</p>", None)])
def test_the_video_a_page_plays(html, vid):
    assert pages.video_id(html, "londonreal.tv") == vid


@pytest.fixture
def page(monkeypatch):
    """A fake fetch serving the page HTML; records what was fetched and the host check."""
    seen = {}

    def download(url, dest, *, allowed, max_bytes, timeout, **kw):
        """Writes the page, or fails as told."""
        seen.update(url=url, allowed=allowed, timeout=timeout)
        if seen.get("fail"):
            raise fetch.FetchError(seen["fail"], "x")
        Path(dest).write_text(seen.get("html", PROMO + PLAYER))
        return {"filename": "", "size": 1}

    monkeypatch.setattr(fetch, "download", download)
    return seen


def test_classify_resolves_the_page_with_a_rebuilt_url(page):
    v = urls.classify(PAGE.replace("londonreal.tv", "www.londonreal.tv") + "?utm=1#x")
    assert (v.platform, v.video_id, v.page_url) == ("youtube", "TeRAr7tE48A", PAGE)
    assert page["url"] == PAGE and page["timeout"] == pages.TIMEOUT  # the bot's own URL, no query
    assert page["allowed"]("londonreal.tv") and not page["allowed"]("evil.com")  # redirects stay on the site


@pytest.mark.parametrize("fail, html, message", [
    ("denied", None, "doesn't exist"), ("failed", None, "Couldn't open"),
    (None, "<title>Join</title>", "members-only")])
def test_page_failures_have_clear_messages(page, fail, html, message):
    page["fail"], page["html"] = fail, html or ""
    with pytest.raises(urls.UnsupportedURL, match=message):
        urls.classify(PAGE)


def _result(page_url="", page_urls=()):
    """A summary result of the episode."""
    summary = {"title": "T", "is_clickbait": False, "clickbait_answer": "", "summary": "S",
               "_stats": {"steps": [], "total": 1, "llm": "x"}}
    r = pipeline.Result("youtube", "TeRAr7tE48A", "https://www.youtube.com/watch?v=TeRAr7tE48A", {"title": "T"},
                        "", "captions", "en", summary)
    r.page_url, r.page_urls = page_url, list(page_urls)
    return r


def test_the_footer_shows_the_page_and_earlier_pages_only_to_who_may_know():
    own = render.render(_result(PAGE, [PAGE]), reveal_cache=False)[-1]
    assert f"🔗 London Real: {PAGE}" in own and "watch?v=TeRAr7tE48A" in own
    later = _result("", [PAGE])  # the plain YouTube link, sent after someone sent the page
    assert PAGE in render.render(later, reveal_cache=True)[-1]  # an admin, or who sent this video before
    assert PAGE not in render.render(later, reveal_cache=False)[-1]  # anyone else: no hint of earlier requests


def test_the_pipeline_remembers_the_page_outside_the_cached_meta(page, monkeypatch):
    monkeypatch.setattr(pipeline, "_run", lambda video, *a, **k: _result())
    r = pipeline.run(PAGE, lambda *a: None)
    assert r.page_url == PAGE and r.page_urls == [PAGE]
    later = pipeline.run("https://youtu.be/TeRAr7tE48A", lambda *a: None)
    assert later.page_url == "" and later.page_urls == [PAGE] and db.video_pages("youtube", "TeRAr7tE48A") == [PAGE]


@pytest.mark.network
def test_the_real_page_plays_the_episode():
    assert pages.resolve("londonreal.tv", PAGE.split("londonreal.tv")[1]) == \
        "https://www.youtube.com/watch?v=TeRAr7tE48A"
