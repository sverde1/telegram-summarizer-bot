"""Documents shared by Google Drive or Dropbox link: recognising the link and downloading the file.

Telegram lets bots download only files up to 20 MB, so bigger books come as a share link. The link the user
sent is never fetched as such: it is parsed into the service's file id (Drive) or share path (Dropbox), and
the download URL is built from that, on the service's own host. Every redirect must stay on that service's
hosts over HTTPS (fetch.download), so a link can't point the bot anywhere else.
"""
import re
import urllib.parse
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from . import config, fetch

GDRIVE_HOSTS = {"drive.google.com", "docs.google.com", "drive.usercontent.google.com"}
DROPBOX_HOSTS = {"www.dropbox.com", "dropbox.com", "dl.dropboxusercontent.com", "dl.dropbox.com"}
_GID = re.compile(r"[A-Za-z0-9_-]{20,200}")
_TOKEN = re.compile(r"[A-Za-z0-9_-]{4,200}")

NOT_SHARED = ("⚠️ I couldn't download that file. Make sure it's shared as \"Anyone with the link\" and send "
              "the link again.")
FOLDER = "⚠️ That's a link to a folder. Send a link to the file itself."
NOT_A_FILE = "⚠️ That link doesn't point to a file I can download."


class LinkError(Exception):
    """A Drive/Dropbox link can't be used. str() is the message for the user; `detail` is for admins."""

    def __init__(self, message: str, detail: str = ""):
        """Stores the user message and a technical detail."""
        super().__init__(message)
        self.detail = detail


@dataclass(frozen=True)
class FileLink:
    """A recognised share link.

    Attributes:
        service: "Google Drive" or "Dropbox".
        download_url: The URL the bot downloads from (built by the bot, not taken from the user).
        name: The file name, when the link shows it (Dropbox), else "".
    """
    service: str
    download_url: str
    name: str = ""


def _gdrive(parts: urllib.parse.SplitResult) -> FileLink:
    """A Google Drive file link → its direct download URL (`confirm=t` skips the "can't scan" warning page)."""
    path, query = parts.path, urllib.parse.parse_qs(parts.query)
    if "/folders/" in path:
        raise LinkError(FOLDER)
    if m := re.match(r"^/document/d/([^/]+)", path):  # a Google Docs document: exported as DOCX
        if _GID.fullmatch(m.group(1)):
            return FileLink("Google Drive", f"https://docs.google.com/document/d/{m.group(1)}/export?format=docx")
        raise LinkError(NOT_A_FILE, path)
    m = re.match(r"^/file/(?:u/\d+/)?d/([^/]+)", path)
    file_id = m.group(1) if m else (query.get("id") or [""])[0]
    if not _GID.fullmatch(file_id or ""):
        raise LinkError(NOT_A_FILE, path)
    return FileLink("Google Drive",
                    f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t")


def _dropbox(parts: urllib.parse.SplitResult) -> FileLink:
    """A Dropbox share link → the same share with `dl=1` (a direct download), rebuilt on www.dropbox.com."""
    segments = [urllib.parse.unquote(s) for s in parts.path.split("/") if s]
    query = urllib.parse.parse_qs(parts.query)
    if segments[:1] == ["sh"] or segments[:2] == ["scl", "fo"]:  # old and new folder links
        raise LinkError(FOLDER)
    if segments[:2] == ["scl", "fi"] and len(segments) == 4:
        key, name, kind = segments[2], segments[3], "scl/fi"
    elif segments[:1] == ["s"] and len(segments) == 3:
        key, name, kind = segments[1], segments[2], "s"
    else:
        raise LinkError(NOT_A_FILE, parts.path)
    rlkey = (query.get("rlkey") or [""])[0]
    if not _TOKEN.fullmatch(key) or (rlkey and not _TOKEN.fullmatch(rlkey)) or "/" in name or not name.strip():
        raise LinkError(NOT_A_FILE, parts.path)
    params = {"dl": "1"} | ({"rlkey": rlkey} if rlkey else {})
    url = (f"https://www.dropbox.com/{kind}/{key}/{urllib.parse.quote(name)}?"
           f"{urllib.parse.urlencode(params)}")
    return FileLink("Dropbox", url, name[:200])


def parse(url: str) -> FileLink | None:
    """Recognises a Google Drive or Dropbox file link.

    Returns:
        The link, or None when it isn't on Google Drive or Dropbox at all (it may be a video link).

    Raises:
        LinkError: A Drive/Dropbox link that doesn't point to a downloadable file (a folder, say).
    """
    try:
        parts = urllib.parse.urlsplit(url.strip())
        parts.port  # noqa: B018  (raises ValueError for a malformed port)
    except ValueError:
        return None
    host = (parts.hostname or "").rstrip(".").lower()
    if host not in GDRIVE_HOSTS | DROPBOX_HOSTS:
        return None
    if parts.scheme not in ("http", "https") or parts.username or parts.password or parts.port not in (None, 443):
        raise LinkError(NOT_A_FILE, "unusual link")
    return _gdrive(parts) if host in GDRIVE_HOSTS else _dropbox(parts)


def _allowed(service: str):
    """The hosts a download from this service may go to (its own, and its file CDN)."""
    if service == "Google Drive":
        return lambda host: host in GDRIVE_HOSTS or host.endswith(".googleusercontent.com")
    return lambda host: host in DROPBOX_HOSTS or host.endswith(".dropboxusercontent.com")


def peek(link: FileLink) -> dict:
    """The name, type and size of a shared file, from the first bytes only (see fetch.peek).

    Returns:
        {"filename", "content_type", "size"}; the name falls back to the one in the link.

    Raises:
        LinkError: Not shared publicly, or the service couldn't be reached.
    """
    try:
        got = fetch.peek(link.download_url, allowed=_allowed(link.service))
    except fetch.FetchError as e:
        if e.code in ("html", "blocked", "denied"):
            raise LinkError(NOT_SHARED, f"{e.code}: {e.detail}")
        raise LinkError("⚠️ Couldn't open that link. Please try again later.", f"{e.code}: {e.detail}")
    got["filename"] = PurePosixPath(got["filename"]).name[:200] or link.name
    return got


def download(link: FileLink, dest: Path, progress=None, max_mb: int | None = None) -> str:
    """Downloads a shared file.

    Args:
        link: From parse().
        dest: Where to save it.
        progress: Called with (bytes, total or None) as the download goes.
        max_mb: Size cap in MB; MAX_LINK_DOWNLOAD_MB (documents) by default.

    Returns:
        The file's name (from the service's answer, else from the link), or "".

    Raises:
        LinkError: Not shared publicly, too large, or the download failed (message for the user).
        proc.ProcCancelled: The job was cancelled.
    """
    max_mb = max_mb or config.MAX_LINK_DOWNLOAD_MB
    limit = max_mb * 1024 ** 2
    try:
        got = fetch.download(link.download_url, dest, allowed=_allowed(link.service), max_bytes=limit,
                             refuse_html=True, progress=progress)
    except fetch.FetchError as e:
        if e.code == "too_large":
            raise LinkError(f"⚠️ This file is larger than {max_mb} MB, the most I download.", e.detail)
        if e.code in ("html", "blocked", "denied"):  # a login page, "access denied", or no such file
            raise LinkError(NOT_SHARED, f"{e.code}: {e.detail}")
        raise LinkError("⚠️ Couldn't download the file. Please try again later.", f"{e.code}: {e.detail}")
    return PurePosixPath(got["filename"]).name[:200] or link.name
