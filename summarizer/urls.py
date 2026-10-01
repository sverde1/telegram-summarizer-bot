"""Find, classify and normalise YouTube / TikTok links."""
import re
import urllib.request
from dataclasses import dataclass

URL_RE = re.compile(r"https?://\S+")

# The 11-character video id is extracted so that every form of a link (watch?v=, shorts/, youtu.be,
# with tracking parameters like ?si= or &pp=) maps to the same cache key.
_YT_PATTERNS = [
    re.compile(r"(?:www\.|m\.|music\.)?youtube\.com/(?:watch\?(?:.*&)?v=|shorts/|live/|embed/)([\w-]{11})"),
    re.compile(r"youtu\.be/([\w-]{11})"),
]
_TT_POST = re.compile(r"tiktok\.com/@([^/?#]+)/(video|photo)/(\d+)")
_TT_SHORT = re.compile(r"(?:vm|vt)\.tiktok\.com/|tiktok\.com/t/")


@dataclass(frozen=True)
class Video:
    """A recognised video link, normalised to platform + id."""

    platform: str  # youtube | tiktok
    video_id: str
    url: str       # canonical URL (TikTok: always the /video/ form, which yt-dlp understands)
    kind: str = "video"  # video | photo (TikTok carousel; also detected later from the probe)


class UnsupportedURL(ValueError):
    """The link isn't a YouTube or TikTok video; the message is shown to the user."""


def _resolve_redirect(url: str) -> str:
    """Follows a TikTok short link (vm./vt.tiktok.com, tiktok.com/t/) to the post URL.

    The short link's code says nothing about the video id, which is needed for the cache key.

    Args:
        url: The short link.

    Returns:
        The final URL after redirects.

    Raises:
        urllib.error.URLError: The link couldn't be resolved.
    """
    # Browser-like headers: TikTok answers bare script requests differently than browsers.
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0",
                                               "Referer": "https://www.tiktok.com/"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.geturl()


def find_url(text: str) -> str | None:
    """Returns the first http(s) URL in a message, or None if there is none."""
    m = URL_RE.search(text or "")
    return m.group(0) if m else None


def classify(url: str) -> Video:
    """Recognises a YouTube or TikTok link and normalises it.

    Args:
        url: A link as the user sent it.

    Returns:
        The platform, video id and canonical URL.

    Raises:
        UnsupportedURL: It's not a YouTube or TikTok video link.
        urllib.error.URLError: A TikTok short link couldn't be resolved.
    """
    for pat in _YT_PATTERNS:
        if m := pat.search(url):
            vid = m.group(1)
            return Video("youtube", vid, f"https://www.youtube.com/watch?v={vid}")

    if _TT_SHORT.search(url):
        url = _resolve_redirect(url)
    if m := _TT_POST.search(url):
        user, kind, vid = m.groups()
        # Carousels are served under /video/ too; yt-dlp rejects the /photo/ form, gallery-dl takes both.
        return Video("tiktok", vid, f"https://www.tiktok.com/@{user}/video/{vid}", kind)

    raise UnsupportedURL("That doesn't look like a YouTube or TikTok video link.")
