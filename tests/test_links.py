"""Documents shared by Google Drive / Dropbox link: recognising links, safe downloads, the upload flow."""
import asyncio
import io
import re
import shutil
import urllib.error
import urllib.request
import urllib.response

import pytest

import access
import bot
from summarizer import config, fetch, links, summarize

from conftest import callback_update, msg_update, send
from docs import make_pdf
from helpers import BookAI

ANA = 60
FILE_ID = "1AbCdEfGhIjKlMnOpQrStUvWxYz0123456"


# ---------- recognising links ----------

@pytest.mark.parametrize("url", [
    f"https://drive.google.com/file/d/{FILE_ID}/view?usp=sharing",
    f"https://drive.google.com/file/u/0/d/{FILE_ID}/edit",
    f"https://drive.google.com/open?id={FILE_ID}",
    f"https://drive.google.com/uc?id={FILE_ID}&export=download",
    f"https://drive.usercontent.google.com/download?id={FILE_ID}",
])
def test_drive_file_links(url):
    link = links.parse(url)
    assert link.service == "Google Drive" and link.download_url == (
        f"https://drive.usercontent.google.com/download?id={FILE_ID}&export=download&confirm=t")


def test_google_docs_document_is_exported_as_docx():
    link = links.parse(f"https://docs.google.com/document/d/{FILE_ID}/edit?usp=sharing")
    assert link.download_url == f"https://docs.google.com/document/d/{FILE_ID}/export?format=docx"


@pytest.mark.parametrize("url,expected,name", [
    ("https://www.dropbox.com/s/abc123xyz/My%20Book.pdf?dl=0",
     "https://www.dropbox.com/s/abc123xyz/My%20Book.pdf?dl=1", "My Book.pdf"),
    ("https://www.dropbox.com/scl/fi/k3y4b5c6d7/book.epub?rlkey=abcdEFGH1234&e=1&dl=0",
     "https://www.dropbox.com/scl/fi/k3y4b5c6d7/book.epub?dl=1&rlkey=abcdEFGH1234", "book.epub"),
])
def test_dropbox_file_links_are_rebuilt(url, expected, name):
    link = links.parse(url)
    assert (link.service, link.download_url, link.name) == ("Dropbox", expected, name)


@pytest.mark.parametrize("url,message", [
    ("https://drive.google.com/drive/folders/1AbCdEfGhIjKlMnOpQrStUvWxYz", links.FOLDER),
    ("https://www.dropbox.com/sh/abc123/AADxyz?dl=0", links.FOLDER),
    ("https://www.dropbox.com/scl/fo/abc123/xyz?rlkey=abcd&dl=0", links.FOLDER),
    ("https://drive.google.com/file/d/short/view", links.NOT_A_FILE),
    ("https://drive.google.com:8443/file/d/" + FILE_ID, links.NOT_A_FILE),
    ("https://www.dropbox.com/home/private", links.NOT_A_FILE),
])
def test_unusable_share_links(url, message):
    with pytest.raises(links.LinkError) as e:
        links.parse(url)
    assert str(e.value) == message


@pytest.mark.parametrize("url", [
    "https://drive.google.com.evil.com/file/d/" + FILE_ID,
    "https://evil.com/drive.google.com/file/d/" + FILE_ID,
    "https://drive.google.com@evil.com/file/d/" + FILE_ID,   # the host is evil.com
    "https://www.youtube.com/watch?v=abcdefghijk",
])
def test_other_links_are_not_share_links(url):
    assert links.parse(url) is None


# ---------- downloading ----------

class _Response(io.BytesIO):
    """A fake HTTP response with headers."""

    def __init__(self, data: bytes, headers: dict):
        """Stores the body and headers."""
        super().__init__(data)
        self.headers = headers

    def __enter__(self):
        """Context manager like urllib's response."""
        return self

    def __exit__(self, *exc):
        """Nothing to close."""


def _serve(monkeypatch, data=b"%PDF-1.4 data", headers=None, error=None):
    """Makes every download answer with this response (or raise `error`); returns the requested URLs."""
    seen = []

    class Opener:
        """Answers any request with the canned response."""

        def open(self, request, timeout):
            """Records the URL, then answers."""
            seen.append(request.full_url)
            if error:
                raise error
            return _Response(data, headers or {"Content-Type": "application/pdf"})

    monkeypatch.setattr(urllib.request, "build_opener", lambda *handlers: Opener())
    return seen


def test_download_uses_the_bot_built_url_and_the_file_name(monkeypatch, tmp_path):
    seen = _serve(monkeypatch, headers={"Content-Type": "application/pdf",
                                        "Content-Disposition": "attachment; filename*=UTF-8''Knji%C5%BEnica.pdf"})
    link = links.parse(f"https://drive.google.com/file/d/{FILE_ID}/view")
    assert links.download(link, tmp_path / "f") == "Knjižnica.pdf"
    assert seen == [link.download_url] and (tmp_path / "f").read_bytes().startswith(b"%PDF")


@pytest.mark.parametrize("serve,message", [
    ({"headers": {"Content-Type": "text/html; charset=utf-8"}}, links.NOT_SHARED),  # a login / denied page
    ({"error": urllib.error.HTTPError("u", 404, "Not Found", {}, None)}, links.NOT_SHARED),
    ({"headers": {"Content-Type": "application/pdf", "Content-Length": str(10 ** 10)}}, "larger than 100 MB"),
])
def test_download_failures(monkeypatch, tmp_path, serve, message):
    _serve(monkeypatch, **serve)
    with pytest.raises(links.LinkError, match=re.escape(message)):
        links.download(links.parse(f"https://drive.google.com/file/d/{FILE_ID}/view"), tmp_path / "f")
    assert not (tmp_path / "f").exists() and not (tmp_path / "f.part").exists()  # nothing half-written


def test_size_cap_while_streaming(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "MAX_LINK_DOWNLOAD_MB", 1)
    _serve(monkeypatch, data=b"x" * (2 * 1024 ** 2))  # no Content-Length: caught while streaming
    with pytest.raises(links.LinkError, match="larger than 1 MB"):
        links.download(links.parse("https://www.dropbox.com/s/abc123xyz/b.pdf"), tmp_path / "f")


def test_redirects_off_the_service_are_refused(tmp_path):
    import http.client
    target = tmp_path / "f"

    class Redirecting(urllib.request.BaseHandler):
        """Answers the first request with a redirect to another host, like a hostile share link could."""

        handler_order = 100  # before urllib's real HTTPS handler (which would go to the network)

        def https_open(self, req):
            """Returns a 302 to evil.com."""
            resp = urllib.response.addinfourl(io.BytesIO(b""), http.client.HTTPMessage(), req.full_url, 302)
            resp.headers["Location"] = "https://evil.com/steal"
            resp.msg = "Found"
            return resp

    real_build = urllib.request.build_opener
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(urllib.request, "build_opener", lambda *handlers: real_build(Redirecting(), *handlers))
        with pytest.raises(fetch.FetchError) as e:
            fetch.download("https://www.dropbox.com/s/abc/b.pdf?dl=1", target,
                           allowed=lambda host: host == "www.dropbox.com", max_bytes=10)
    assert e.value.code == "blocked" and "evil.com" in e.value.detail and not target.exists()


@pytest.mark.parametrize("url", ["https://evil.com/x", "http://www.dropbox.com/x"])
def test_only_https_on_allowed_hosts(url, tmp_path):
    with pytest.raises(fetch.FetchError) as e:
        fetch.download(url, tmp_path / "f", allowed=lambda host: host == "www.dropbox.com", max_bytes=10)
    assert e.value.code == "blocked"


# ---------- in the bot ----------

@pytest.fixture
def served_pdf(monkeypatch, tmp_path):
    """Approves ANA, fakes the AI, and makes every link download serve a small book."""
    access.set_state(ANA, "allowed")
    BookAI.calls = []
    monkeypatch.setattr(summarize, "conversation", lambda backend=None, model=None: BookAI())
    pdf = make_pdf(tmp_path / "src.pdf", ["Intro text about cats and their habits " * 3,
                                          "Second part on dogs and their loyalty " * 3], [("Cats", 0), ("Dogs", 1)])

    def download(link, dest, progress=None):
        """Copies the book; reports the name a service would."""
        shutil.copy(pdf, dest)
        if progress:
            progress(1000, 2000)
        return link.name or "Real name.pdf"

    monkeypatch.setattr(links, "download", download)


async def _run(app):
    """Runs the worker until the queue is empty."""
    task = asyncio.create_task(bot.worker(app))
    await asyncio.wait_for(bot.queue.join(), 20)
    task.cancel()


async def test_dropbox_link_is_summarized_like_an_upload(app, telegram, served_pdf):
    await send(app, msg_update(ANA, "look https://www.dropbox.com/s/abc123xyz/Pets.pdf?dl=0"))
    offer = telegram.sent("sendMessage")[-1]
    assert offer["text"].startswith("📄 Pets.pdf (Dropbox)\nHow should I summarize it?")
    up = int(re.search(r"book:(\d+):", str(offer["reply_markup"])).group(1))
    await send(app, callback_update(ANA, f"book:{up}:whole"))
    await _run(app)
    assert "Whole book." in telegram.sent("sendMessage")[-1]["text"]
    assert any("📥 Downloading the file… 1 KB of 2 KB" in t for t in telegram.texts())


async def test_drive_link_learns_the_file_name_on_download(app, telegram, served_pdf):
    await send(app, msg_update(ANA, f"https://drive.google.com/file/d/{FILE_ID}/view?usp=sharing"))
    assert telegram.sent("sendMessage")[-1]["text"].startswith("📄 Google Drive file (Google Drive)")
    up = int(re.search(r"book:(\d+):", str(telegram.sent("sendMessage")[-1]["reply_markup"])).group(1))
    await send(app, callback_update(ANA, f"book:{up}:whole"))
    await _run(app)
    assert "Real name.pdf" in telegram.sent("sendMessage")[-1]["text"]


@pytest.mark.parametrize("url,reply", [
    ("https://www.dropbox.com/s/abc123xyz/book.mobi?dl=0", "Convert it to PDF or EPUB"),
    ("https://www.dropbox.com/s/abc123xyz/song.mp3?dl=0", "I can summarize PDF, EPUB, DOCX or TXT"),
    ("https://drive.google.com/drive/folders/1AbCdEfGhIjKlMnOpQrStUvWxYz", links.FOLDER),
])
async def test_unusable_links_are_refused_at_once(app, telegram, url, reply):
    access.set_state(ANA, "allowed")
    await send(app, msg_update(ANA, url))
    assert reply in telegram.texts()[-1] and bot.queue.qsize() == 0


def test_too_big_upload_points_to_drive_and_dropbox():
    assert "Google Drive or Dropbox" in bot.TOO_BIG
