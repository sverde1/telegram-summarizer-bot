"""Video pages: sites whose episode pages play an embedded YouTube video (London Real for now).

Only the sites in SITES are ever fetched, by a URL the bot rebuilds from the link's host and path (never the
user's string), through fetch.download with every redirect checked against SITES. The page's HTML is untrusted:
it's only scanned with regular expressions for the YouTube video it declares, never parsed or run, and never
logged. A compromised page could at most make the bot summarize a different public YouTube video.
"""
import logging
import re
import tempfile
from pathlib import Path

from . import fetch

log = logging.getLogger(__name__)

SITES = {"londonreal.tv": "London Real", "www.londonreal.tv": "London Real"}  # host -> name in messages
_SLUG = re.compile(r"/[a-z0-9]+(?:-[a-z0-9]+)*/?", re.I)  # an episode page: /some-episode-title/
MAX_BYTES = 2 * 1024 ** 2  # an episode page is about 650 KB
TIMEOUT = 20  # seconds without data
_ID = r"([\w-]{11})"
# The order of trust: the page's own structured data, its main player, else its first embedded video.
_JSON_LD = re.compile(r'"embedUrl"\s*:\s*"https?:\\?/\\?/(?:www\.)?youtube(?:-nocookie)?\.com\\?/embed\\?/' + _ID)
_EMBED = re.compile(r"youtube(?:-nocookie)?\.com\\?/embed\\?/" + _ID + r"([^\"'\s<>]*)")


class PageError(Exception):
    """A page link that can't be used; str() is the message for the user, `detail` is for the log."""

    def __init__(self, message: str, detail: str = ""):
        """Stores the user message and a technical detail."""
        super().__init__(message)
        self.detail = detail


def is_page(host: str) -> bool:
    """Whether a (lowercased) host is a supported video-page site."""
    return host in SITES


def is_episode(host: str, path: str) -> bool:
    """Whether a link points to an episode page (a slug), checked without any network access."""
    return is_page(host) and bool(_SLUG.fullmatch(path))


def name(host: str) -> str:
    """The site's name for messages, e.g. "London Real"."""
    return SITES.get(host, host)


def page_url(host: str, path: str) -> str:
    """The URL the bot fetches and shows: rebuilt from the site's main host and the path (no query)."""
    return f"https://{host.removeprefix('www.')}{path}"


def video_id(html: str, host: str) -> str | None:
    """The YouTube video an episode page declares: its JSON-LD embedUrl, else the main player's embed (the one
    with origin=<the site>), else the first embedded video."""
    if m := _JSON_LD.search(html):
        return m.group(1)
    embeds = [(e.group(1), e.group(2).replace("&amp;", "&")) for e in _EMBED.finditer(html)]
    origin = f"origin=https://{host.removeprefix('www.')}"
    for vid, query in embeds:
        if origin in query:
            return vid
    return embeds[0][0] if embeds else None


def resolve(host: str, path: str) -> str:
    """The YouTube watch URL of the video an episode page plays (blocking: run it in the worker thread).

    Raises:
        PageError: The page doesn't exist, couldn't be opened, or has no YouTube video (members-only).
    """
    site = name(host)
    url = page_url(host, path)
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "page.html"
        try:
            fetch.download(url, dest, allowed=is_page, max_bytes=MAX_BYTES, timeout=TIMEOUT)
        except fetch.FetchError as e:
            if e.code == "denied":
                raise PageError(f"That {site} page doesn't exist.", f"{e.code}: {e.detail}")
            raise PageError(f"Couldn't open that {site} page. Try again later.", f"{e.code}: {e.detail}")
        html = dest.read_bytes().decode("utf-8", "replace")
    vid = video_id(html, host)
    if not vid:
        title = re.search(r"<title>([^<]{0,200})", html)
        raise PageError(f"Couldn't find a video on that {site} page (members-only episodes can't be read).",
                        f"no YouTube embed; title: {title.group(1) if title else '?'}")
    log.info("%s page %s plays YouTube video %s", site, path[:80], vid)
    return f"https://www.youtube.com/watch?v={vid}"
