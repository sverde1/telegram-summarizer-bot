"""/ocrlang: listing, adding and removing OCR languages, with downloads only from fixed, checked sources."""
import io
import shutil

import pytest

import access
from summarizer import config, db, ocr

from conftest import ADMIN_ID, msg_update, send
from tgbot import jobs, state

pytestmark = pytest.mark.skipif(not ocr.available_tesseract(), reason="Tesseract isn't installed")


@pytest.fixture
def fake_download(monkeypatch):
    """Downloads "succeed" with the system's English model (a valid Tesseract model under another name)."""
    urls = []

    def download(url, dest, sha256=None):
        """Records the URL and writes a working model."""
        urls.append(url)
        shutil.copy(ocr.tessdata() / "eng.traineddata", dest)

    monkeypatch.setattr(ocr, "_download", download)
    return urls


async def test_only_admins_get_an_answer(app, telegram):
    access.set_state(60, "allowed")
    await send(app, msg_update(60, "/ocrlang add slv"))
    assert telegram.texts() == []


async def test_list(app, telegram):
    await send(app, msg_update(ADMIN_ID, "/ocrlang"))
    text = telegram.texts()[-1]
    assert "OCR engine: Tesseract" in text and "Installed: English (eng)" in text and "slv Slovenian" in text


async def test_add_validates_and_installs(app, telegram, fake_download):
    await send(app, msg_update(ADMIN_ID, "/ocrlang add slv"))
    assert telegram.texts()[-2] == "⏳ Downloading the Slovenian model…"
    assert telegram.texts()[-1] == "✅ Slovenian (slv) added. Text recognition now reads: English, Slovenian."
    assert fake_download == ["https://github.com/tesseract-ocr/tessdata_fast/raw/main/slv.traineddata"]
    assert ocr.installed() == ["eng", "slv"] and (ocr.tessdata() / "slv.traineddata").exists()


async def test_unknown_code_is_refused_without_a_download(app, telegram):
    await send(app, msg_update(ADMIN_ID, "/ocrlang add ../../etc"))  # the network is blocked in tests anyway
    assert telegram.texts()[-1].startswith("⚠️ Unknown language code") and ocr.installed() == ["eng"]


async def test_a_broken_model_is_not_installed(app, telegram, monkeypatch):
    monkeypatch.setattr(ocr, "_download", lambda url, dest, sha256=None: dest.write_bytes(b"not a model"))
    await send(app, msg_update(ADMIN_ID, "/ocrlang add deu"))
    assert telegram.texts()[-1].startswith("⚠️ Tesseract can't use the downloaded model")
    assert ocr.installed() == ["eng"] and not (ocr.tessdata() / "deu.traineddata").exists()


async def test_remove(app, telegram, fake_download):
    await send(app, msg_update(ADMIN_ID, "/ocrlang remove eng"))
    assert "only language" in telegram.texts()[-1]
    await send(app, msg_update(ADMIN_ID, "/ocrlang add slv"))
    await send(app, msg_update(ADMIN_ID, "/ocrlang remove slv"))
    assert telegram.texts()[-1].startswith("✅ Slovenian (slv) removed") and ocr.installed() == ["eng"]
    assert not (ocr.tessdata() / "slv.traineddata").exists()


def test_rapidocr_languages_need_their_script_model(monkeypatch):
    monkeypatch.setattr(config, "OCR_ENGINE", "rapidocr")
    with pytest.raises(ocr.LanguageError, match="RapidOCR can't read Greek"):
        ocr.add_language("ell")
    db.set_setting("ocr_languages", "eng,slv")
    assert ocr.installed() == ["eng"]  # Slovenian needs the Latin model, which isn't installed


class _Response(io.BytesIO):
    """A fake HTTP response."""

    headers = {"Content-Length": "4"}

    def __enter__(self):
        """Context manager like urllib's response."""
        return self

    def __exit__(self, *exc):
        """Nothing to close."""


def test_download_checks_the_checksum(monkeypatch, tmp_path):
    import urllib.request

    class Opener:
        """Serves 4 bytes for any URL."""

        def open(self, url, timeout):
            """Returns the fake response."""
            return _Response(b"data")

    monkeypatch.setattr(urllib.request, "build_opener", lambda *handlers: Opener())
    with pytest.raises(ocr.LanguageError, match="checksum"):
        ocr._download("https://www.modelscope.cn/x.onnx", tmp_path / "x.onnx", "0" * 64)
    assert not (tmp_path / "x.onnx").exists() and not (tmp_path / "x.onnx.part").exists()


async def test_admins_are_told_how_to_add_a_missing_language(app, telegram, tmp_path, monkeypatch):
    import re
    from summarizer import summarize
    from conftest import callback_update, doc_update
    from docs import make_scan
    from helpers import BookAI

    def refuse(*a, **k):
        """The sample looks Polish."""
        raise ocr.UnsupportedLanguage("⚠️ This scan looks like Polish, which text recognition doesn't support "
                                      "yet. Supported: English.", "pol")

    monkeypatch.setattr(ocr, "choose_languages", refuse)
    monkeypatch.setattr(summarize, "conversation", lambda backend=None, model=None: BookAI())
    scan = make_scan(tmp_path / "s.pdf", ["Tekst " * 30])
    telegram.files["F1"] = scan
    await send(app, doc_update(ADMIN_ID, "s.pdf", scan.stat().st_size))
    up = int(re.search(r"book:(\d+):", str(telegram.sent("sendMessage")[-1]["reply_markup"])).group(1))
    await send(app, callback_update(ADMIN_ID, f"book:{up}:whole"))
    import asyncio
    import bot
    task = asyncio.create_task(jobs.worker(app))
    await asyncio.wait_for(state.queue.join(), 30)
    task.cancel()
    assert telegram.texts()[-1].endswith("Details: add it with /ocrlang add pol")
