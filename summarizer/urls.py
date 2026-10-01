"""Find, check and normalise YouTube / TikTok links.

Only YouTube and TikTok are supported, and links are checked by their parsed host name, never by searching
the text: a check like "contains vm.tiktok.com/" let `http://192.168.1.1/?vm.tiktok.com/` through, and the
bot then fetched an address on its own network (SSRF). The only network request made here is resolving a
TikTok short link, and every redirect hop of it must stay on TikTok over HTTPS.
"""
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

URL_RE = re.compile(r"https?://\S+")

YOUTUBE_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be",
                 "www.youtu.be", "youtube-nocookie.com", "www.youtube-nocookie.com"}
TIKTOK_POST_HOSTS = {"tiktok.com", "www.tiktok.com", "m.tiktok.com"}
TIKTOK_SHORT_HOSTS = {"vm.tiktok.com", "vt.tiktok.com"}
TIKTOK_HOSTS = TIKTOK_POST_HOSTS | TIKTOK_SHORT_HOSTS

# The 11-character video id is extracted so that every form of a link (watch?v=, shorts/, youtu.be, with
# tracking parameters like ?si= or &pp=) maps to the same cache key.
_YT_ID = re.compile(r"[\w-]{11}")
_YT_PATH = re.compile(r"/(?:shorts|live|embed)/([\w-]{11})/?")
_TT_POST = re.compile(r"/@([^/?#]+)/(video|photo)/(\d+)/?")
_TT_SHORT_PATH = re.compile(r"/[A-Za-z0-9_-]+/?")      # vm./vt.tiktok.com/<code>
_TT_T_PATH = re.compile(r"/t/[A-Za-z0-9_-]+/?")        # tiktok.com/t/<code>

MAX_REDIRECTS = 5
RESOLVE_TIMEOUT = 10  # seconds; the short-link lookup happens while the user waits

NOT_SUPPORTED = "Only YouTube and TikTok links are supported."
NOT_A_YOUTUBE_VIDEO = "That YouTube link doesn't point to a video."
NOT_A_TIKTOK_POST = "That TikTok link doesn't point to a video or photo post."
SHORT_LINK_FAILED = "Couldn't open that TikTok short link. Try the full link from the TikTok app's Share menu."


@dataclass(frozen=True)
class Video:
    """A recognised video link, normalised to platform + id."""

    platform: str  # youtube | tiktok
    video_id: str
    url: str       # canonical URL (TikTok: always the /video/ form, which yt-dlp understands)
    kind: str = "video"  # video | photo (TikTok carousel; also detected later from the probe)


class UnsupportedURL(ValueError):
    """The link isn't a supported YouTube or TikTok link; the message is shown to the user."""


@dataclass(frozen=True)
class _Parsed:
    """A link split into the parts the checks need."""

    host: str
    path: str
    query: str


def _parse(url: str) -> _Parsed:
    """Splits a link and rejects anything that isn't a plain web link to YouTube or TikTok.

    Raises:
        UnsupportedURL: Not http(s), has a user name/password (`youtube.com@evil.com`), an unusual port, or a
            host outside YouTube/TikTok.
    """
    try:
        parts = urllib.parse.urlsplit(url.strip())
        port = parts.port  # raises ValueError for a malformed port
    except ValueError:
        raise UnsupportedURL(NOT_SUPPORTED)
    host = (parts.hostname or "").rstrip(".").lower()  # "youtube.com." is the same host
    if (parts.scheme not in ("http", "https") or parts.username is not None or parts.password is not None
            or port not in (None, 80, 443) or host not in YOUTUBE_HOSTS | TIKTOK_HOSTS):
        raise UnsupportedURL(NOT_SUPPORTED)
    return _Parsed(host, parts.path, parts.query)


def _youtube_id(p: _Parsed) -> str | None:
    """The 11-character video id of a YouTube link, or None if it isn't a video link."""
    if p.host in ("youtu.be", "www.youtu.be"):
        m = re.fullmatch(r"/([\w-]{11})/?", p.path)
        return m.group(1) if m else None
    if p.path.rstrip("/") == "/watch":
        v = urllib.parse.parse_qs(p.query).get("v", [""])[0]
        return v if _YT_ID.fullmatch(v) else None
    m = _YT_PATH.fullmatch(p.path)
    return m.group(1) if m else None


def _tiktok_video(p: _Parsed) -> "Video | None":
    """The Video for a full TikTok post link (/@user/video/<id> or /@user/photo/<id>), else None."""
    if p.host not in TIKTOK_POST_HOSTS:
        return None
    m = _TT_POST.fullmatch(p.path)
    if not m:
        return None
    user, kind, vid = m.groups()
    # Carousels are served under /video/ too; yt-dlp rejects the /photo/ form, gallery-dl takes both.
    return Video("tiktok", vid, f"https://www.tiktok.com/@{user}/video/{vid}", kind)


def _is_short_link(p: _Parsed) -> bool:
    """Whether this is a TikTok short link (vm./vt.tiktok.com/<code> or tiktok.com/t/<code>)."""
    return ((p.host in TIKTOK_SHORT_HOSTS and bool(_TT_SHORT_PATH.fullmatch(p.path)))
            or (p.host in TIKTOK_POST_HOSTS and bool(_TT_T_PATH.fullmatch(p.path))))


def find_url(text: str) -> str | None:
    """Returns the first http(s) link in a message, or None."""
    m = URL_RE.search(text or "")
    return m.group(0) if m else None


def check(url: str) -> None:
    """Checks a link without any network access (used before a link is queued).

    Raises:
        UnsupportedURL: With a message for the user saying what's wrong.
    """
    p = _parse(url)
    if p.host in YOUTUBE_HOSTS:
        if not _youtube_id(p):
            raise UnsupportedURL(NOT_A_YOUTUBE_VIDEO)
    elif not (_tiktok_video(p) or _is_short_link(p)):
        raise UnsupportedURL(NOT_A_TIKTOK_POST)


class _FoundPost(Exception):
    """Raised inside the redirect handler as soon as a hop points at a TikTok post."""

    def __init__(self, video: Video):
        """Carries the post found."""
        super().__init__(video.url)
        self.video = video


class _TikTokRedirects(urllib.request.HTTPRedirectHandler):
    """Follows short-link redirects only within TikTok over HTTPS, and stops at the first post link.

    Stopping early avoids fetching the TikTok page itself, which is slow and subject to bot detection.
    """

    max_redirections = MAX_REDIRECTS

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        """Validates one redirect hop before it's followed.

        Raises:
            _FoundPost: The hop points at a TikTok post (resolution is done).
            UnsupportedURL: The hop leaves TikTok, or isn't HTTPS.
        """
        newurl = urllib.parse.urljoin(req.full_url, newurl)
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise UnsupportedURL(SHORT_LINK_FAILED)
        try:
            p = _parse(newurl)
        except UnsupportedURL:
            raise UnsupportedURL(SHORT_LINK_FAILED)
        if p.host not in TIKTOK_HOSTS:
            raise UnsupportedURL(SHORT_LINK_FAILED)
        if video := _tiktok_video(p):
            raise _FoundPost(video)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _resolve_short_link(p: _Parsed) -> Video:
    """Follows a TikTok short link to the post it points at.

    The short code says nothing about the video id, which is needed for the cache key.

    Raises:
        UnsupportedURL: The link didn't lead to a TikTok post.
    """
    opener = urllib.request.build_opener(_TikTokRedirects())
    # Always HTTPS, and only host + path: the original scheme, port or query don't matter for a short link.
    req = urllib.request.Request(f"https://{p.host}{p.path}",
                                 # Browser-like headers: TikTok answers bare script requests differently.
                                 headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.tiktok.com/"})
    try:
        with opener.open(req, timeout=RESOLVE_TIMEOUT) as r:
            final = r.geturl()
    except _FoundPost as found:
        return found.video
    except (urllib.error.URLError, OSError, ValueError):
        raise UnsupportedURL(SHORT_LINK_FAILED)
    try:
        video = _tiktok_video(_parse(final))
    except UnsupportedURL:
        video = None
    if not video:
        raise UnsupportedURL(SHORT_LINK_FAILED)
    return video


def classify(url: str) -> Video:
    """Turns a supported link into a Video (platform, id, canonical URL).

    TikTok short links are resolved over the network; call this from the worker thread, not the event loop.

    Raises:
        UnsupportedURL: Not a YouTube video or TikTok post link, or a short link that couldn't be resolved.
    """
    check(url)
    p = _parse(url)
    if p.host in YOUTUBE_HOSTS:
        vid = _youtube_id(p)
        return Video("youtube", vid, f"https://www.youtube.com/watch?v={vid}")
    return _tiktok_video(p) or _resolve_short_link(p)
