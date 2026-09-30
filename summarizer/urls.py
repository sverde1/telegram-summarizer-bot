"""Find, classify and normalise YouTube / TikTok links."""
import re
import urllib.request
from dataclasses import dataclass

URL_RE = re.compile(r"https?://\S+")

_YT_PATTERNS = [
    re.compile(r"(?:www\.|m\.|music\.)?youtube\.com/(?:watch\?(?:.*&)?v=|shorts/|live/|embed/)([\w-]{11})"),
    re.compile(r"youtu\.be/([\w-]{11})"),
]
_TT_POST = re.compile(r"tiktok\.com/@([^/?#]+)/(video|photo)/(\d+)")
_TT_SHORT = re.compile(r"(?:vm|vt)\.tiktok\.com/|tiktok\.com/t/")


@dataclass(frozen=True)
class Video:
    platform: str  # youtube | tiktok
    video_id: str
    url: str       # canonical URL (TikTok: always the /video/ form, which yt-dlp understands)
    kind: str = "video"  # video | photo (TikTok carousel; also detected later from the probe)


class UnsupportedURL(ValueError):
    pass


def _resolve_redirect(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0",
                                               "Referer": "https://www.tiktok.com/"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.geturl()


def find_url(text: str) -> str | None:
    m = URL_RE.search(text or "")
    return m.group(0) if m else None


def classify(url: str) -> Video:
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
