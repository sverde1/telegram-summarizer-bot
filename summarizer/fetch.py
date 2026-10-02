"""Downloads from a fixed set of hosts, safely: HTTPS only, redirects checked hop by hop, size-capped.

Used for files the bot fetches itself (OCR models, documents shared by Google Drive / Dropbox link). The
caller decides which hosts are allowed; a redirect anywhere else stops the download, so a link can't make the
bot fetch from its own network (SSRF) or from an arbitrary server.
"""
import email.message
import hashlib
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

from . import proc

CHUNK = 1 << 16
TIMEOUT = 60  # seconds without data before giving up


class FetchError(Exception):
    """A download failed. `code`: blocked (redirect off the allowed hosts), too_large, html (a web page
    instead of the file: usually a login or "not shared" page), denied (HTTP 401/403/404: private or gone),
    checksum, failed."""

    def __init__(self, code: str, detail: str = ""):
        """Stores the reason code and a technical detail."""
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def download(url: str, dest: Path, *, allowed: Callable[[str], bool], max_bytes: int, sha256: str | None = None,
             refuse_html: bool = False, progress: Callable[[int, int | None], None] | None = None) -> dict:
    """Downloads `url` to `dest` (written atomically), following redirects only to allowed hosts.

    Args:
        url: An https URL on an allowed host.
        dest: Where the file goes.
        allowed: Whether a host name may be contacted (checked for the URL and every redirect).
        max_bytes: Size cap; checked against Content-Length and while streaming.
        sha256: Expected checksum, if known.
        refuse_html: Treat an HTML answer as an error (file hosts answer with a page when access is denied).
        progress: Called with (bytes so far, total or None) about every 2 MB.

    Returns:
        {"filename": name from Content-Disposition or "", "size": bytes}.

    Raises:
        FetchError: See the codes on FetchError.
        proc.ProcCancelled: The current job was cancelled (checked between chunks).
    """

    def check(target: str) -> None:
        """Refuses a URL that isn't https on an allowed host."""
        parts = urllib.parse.urlsplit(target)
        if parts.scheme != "https" or not allowed((parts.hostname or "").lower()):
            raise FetchError("blocked", f"not allowed: {parts.hostname}")

    class Checked(urllib.request.HTTPRedirectHandler):
        """Follows a redirect only to an allowed host over HTTPS."""

        def redirect_request(self, req, fp, code, msg, headers, newurl):
            """Checks the redirect target before following it."""
            check(urllib.parse.urljoin(req.full_url, newurl))
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    check(url)
    tmp = dest.with_name(dest.name + ".part")
    digest, size = hashlib.sha256(), 0
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.build_opener(Checked()).open(request, timeout=TIMEOUT) as r:
            headers = r.headers
            if refuse_html and "text/html" in (headers.get("Content-Type") or ""):
                raise FetchError("html", headers.get("Content-Type") or "")
            total = int(headers.get("Content-Length") or 0) or None
            if total and total > max_bytes:
                raise FetchError("too_large", str(total))
            next_report = 0
            with tmp.open("wb") as f:
                while chunk := r.read(CHUNK):
                    proc.check_cancelled()
                    size += len(chunk)
                    if size > max_bytes:
                        raise FetchError("too_large", f"over {max_bytes}")
                    digest.update(chunk)
                    f.write(chunk)
                    if progress and size >= next_report:
                        progress(size, total)
                        next_report = size + 2 * 1024 ** 2
    except urllib.error.HTTPError as e:
        tmp.unlink(missing_ok=True)
        raise FetchError("denied" if e.code in (401, 403, 404) else "failed", f"HTTP {e.code}")
    except (urllib.error.URLError, OSError, ValueError) as e:
        tmp.unlink(missing_ok=True)
        raise FetchError("failed", str(e))
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    if sha256 and digest.hexdigest() != sha256:
        tmp.unlink(missing_ok=True)
        raise FetchError("checksum")
    tmp.replace(dest)
    disposition = email.message.Message()
    disposition["content-disposition"] = headers.get("Content-Disposition") or ""
    return {"filename": disposition.get_filename() or "", "size": size}


def peek(url: str, *, allowed: Callable[[str], bool], max_bytes: int = 65536, timeout: float = 10) -> dict:
    """Looks at the start of a file without downloading it: its name, type and total size.

    A ranged request (bytes 0-65535); a server that ignores the range answers with the whole file, of which
    at most `max_bytes` are read. Same host and redirect checks as download().

    Returns:
        {"filename": "" if unknown, "content_type": str, "size": total bytes or None}.

    Raises:
        FetchError: blocked, html (a page instead of the file), denied or failed.
    """
    def check(target: str) -> None:
        """Refuses a URL that isn't https on an allowed host."""
        parts = urllib.parse.urlsplit(target)
        if parts.scheme != "https" or not allowed((parts.hostname or "").lower()):
            raise FetchError("blocked", f"not allowed: {parts.hostname}")

    class Checked(urllib.request.HTTPRedirectHandler):
        """Follows a redirect only to an allowed host over HTTPS."""

        def redirect_request(self, req, fp, code, msg, headers, newurl):
            """Checks the redirect target before following it."""
            check(urllib.parse.urljoin(req.full_url, newurl))
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    check(url)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes=0-{max_bytes - 1}"})
    try:
        with urllib.request.build_opener(Checked()).open(request, timeout=timeout) as r:
            headers = r.headers
            r.read(max_bytes)
    except urllib.error.HTTPError as e:
        raise FetchError("denied" if e.code in (401, 403, 404) else "failed", f"HTTP {e.code}")
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise FetchError("failed", str(e))
    content_type = (headers.get("Content-Type") or "").split(";")[0].strip().lower()
    if content_type == "text/html":
        raise FetchError("html", content_type)
    size = None
    if m := re.match(r"bytes \d+-\d+/(\d+)", headers.get("Content-Range") or ""):  # 206: the total after "/"
        size = int(m[1])
    elif (headers.get("Content-Length") or "").isdigit() and not headers.get("Content-Range"):
        size = int(headers["Content-Length"])  # 200: the range was ignored, this is the whole file
    disposition = email.message.Message()
    disposition["content-disposition"] = headers.get("Content-Disposition") or ""
    return {"filename": disposition.get_filename() or "", "content_type": content_type, "size": size}
