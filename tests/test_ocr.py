"""Scanned documents: confirmation, OCR (real Tesseract on generated scans), limits, approval, languages."""
import asyncio
import re

import pytest

import access
import bot
from summarizer import config, db, documents, ocr, proc, summarize

from conftest import ADMIN_ID, callback_update, doc_update, msg_update, send
from docs import make_scan
from helpers import BookAI

ANA, BOB = 60, 61
TEXT = "The quick brown fox jumps over the lazy dog.\nCats sleep most of the day and hunt at night.\n" * 4

pytestmark = pytest.mark.skipif(not ocr.available_tesseract(), reason="Tesseract isn't installed")


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    """Approved users, a fake AI, and two OCR workers."""
    for uid in (ANA, BOB):
        access.set_state(uid, "allowed")
    BookAI.calls = []
    monkeypatch.setattr(summarize, "conversation", lambda backend=None, model=None: BookAI())
    monkeypatch.setattr(config, "OCR_WORKERS", 2)


async def _upload(app, telegram, uid, path, file_id="F1") -> int:
    """Sends a file; returns its upload id."""
    telegram.files[file_id] = path
    await send(app, doc_update(uid, path.name, path.stat().st_size, file_id=file_id))
    return int(re.search(r"book:(\d+):", str(telegram.sent("sendMessage")[-1]["reply_markup"])).group(1))


async def _work(app):
    """Runs the worker until the queue is empty."""
    task = asyncio.create_task(bot.worker(app))
    await asyncio.wait_for(bot.queue.join(), 60)
    task.cancel()


def _scan(tmp_path, pages=2, name="scan.pdf"):
    """A scanned PDF with readable English text on every page."""
    return make_scan(tmp_path / name, [TEXT] * pages)


def _rid(telegram) -> int:
    """The request id in the last OCR confirmation's buttons."""
    return int(re.search(r"ocr:(\d+):", str(telegram.sent("editMessageText")[-1]["reply_markup"])).group(1))


async def test_scan_asks_first_then_reads_and_summarizes(app, telegram, tmp_path):
    up = await _upload(app, telegram, ANA, _scan(tmp_path))
    await send(app, callback_update(ANA, f"book:{up}:whole"))
    await _work(app)
    ask = telegram.sent("editMessageText")[-1]
    assert ask["text"].startswith("🔍 This is a scanned document: 2 pages need text recognition (OCR), about")
    assert "Start OCR" in str(ask["reply_markup"]) and BookAI.calls == []
    assert db.recent_requests(ANA)[0]["status"] == "waiting" and db.ocr_usage(ANA)[0] == 0
    await send(app, callback_update(ANA, f"ocr:{_rid(telegram)}:go"))
    await _work(app)
    text = telegram.sent("sendMessage")[-1]["text"]
    assert "Whole book." in text and "OCR (Tesseract, English)" in text and "OCR " in text
    assert db.ocr_usage(ANA)[0] == 1 and db.daily_usage(ANA)[0] == 1  # one request, counted once each
    pages = db.get_pages(db.get_upload(up)["sha256"])
    assert "quick brown fox" in pages[0] and "quick brown fox" in pages[1]
    assert "quick brown fox" in BookAI.calls[0].text


async def test_cancel_at_the_confirmation(app, telegram, tmp_path):
    up = await _upload(app, telegram, ANA, _scan(tmp_path, 1))
    await send(app, callback_update(ANA, f"book:{up}:whole"))
    await _work(app)
    await send(app, callback_update(ANA, f"ocr:{_rid(telegram)}:no"))
    assert telegram.texts()[-1] == "✖️ Cancelled." and db.recent_requests(ANA)[0]["status"] == "cancelled"
    await send(app, callback_update(ANA, f"ocr:{db.recent_requests(ANA)[0]['id']}:go"))  # too late now
    assert telegram.sent("answerCallbackQuery")[-1]["text"] == "This request isn't waiting any more."


async def test_ocr_limit(app, telegram, tmp_path):
    db.set_setting("ocr_limit", "1")
    old = db.add_request(ANA, "x", "book")
    db.update_request(old, ocr=1, status="done")
    up = await _upload(app, telegram, ANA, _scan(tmp_path, 1))
    await send(app, callback_update(ANA, f"book:{up}:whole"))
    await _work(app)
    assert telegram.texts()[-1].startswith("⏳ This is a scanned document and needs text recognition (OCR). "
                                           "You've used today's 1 OCR documents")
    assert db.recent_requests(ANA)[0]["status"] == "failed"


async def test_over_the_page_cap_an_admin_can_allow_it(app, telegram, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OCR_MAX_PAGES", 1)
    up = await _upload(app, telegram, ANA, _scan(tmp_path, 2))
    await send(app, callback_update(ANA, f"book:{up}:whole"))
    await _work(app)
    ask = telegram.sent("editMessageText")[-1]
    assert "I read up to 1 pages" in ask["text"] and "Ask admin for approval" in str(ask["reply_markup"])
    rid = _rid(telegram)
    await send(app, callback_update(ANA, f"ocr:{rid}:go"))  # forged Start: refused
    assert "admin's approval" in telegram.sent("answerCallbackQuery")[-1]["text"]
    await send(app, callback_update(ANA, f"ocr:{rid}:ask"))
    to_admin = [m for m in telegram.sent("sendMessage") if m["chat_id"] == ADMIN_ID][-1]
    assert "asks to read a long scan" in to_admin["text"] and "Pages needing OCR: 2 (limit 1)" in to_admin["text"]
    assert "Language: English" in to_admin["text"] and "Estimated OCR time" in to_admin["text"]
    await send(app, callback_update(ANA, f"ocr:{rid}:ask"))
    assert telegram.sent("answerCallbackQuery")[-1]["text"] == "An admin has been asked already."
    await send(app, callback_update(BOB, f"ocradm:{rid}:yes"))  # not an admin
    assert telegram.sent("answerCallbackQuery")[-1]["text"] == "Only admins can do this."
    await send(app, callback_update(ADMIN_ID, f"ocradm:{rid}:yes"))
    to_user = telegram.sent("sendMessage")[-1]
    assert to_user["chat_id"] == ANA and "allowed reading all 2 pages" in to_user["text"]
    await send(app, callback_update(ANA, f"ocr:{rid}:go"))
    await _work(app)
    assert "Whole book." in telegram.sent("sendMessage")[-1]["text"]


async def test_admin_can_deny_a_long_scan(app, telegram, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OCR_MAX_PAGES", 1)
    up = await _upload(app, telegram, ANA, _scan(tmp_path, 2))
    await send(app, callback_update(ANA, f"book:{up}:whole"))
    await _work(app)
    rid = _rid(telegram)
    await send(app, callback_update(ANA, f"ocr:{rid}:ask"))
    await send(app, callback_update(ADMIN_ID, f"ocradm:{rid}:no"))
    assert telegram.sent("sendMessage")[-1]["text"] == "❌ An admin declined reading this long scan."
    assert db.get_request(rid)["status"] == "cancelled"


async def test_cached_ocr_text_needs_no_confirmation_and_costs_nothing(app, telegram, tmp_path):
    scan = _scan(tmp_path, 1)
    up = await _upload(app, telegram, ADMIN_ID, scan)
    await send(app, callback_update(ADMIN_ID, f"book:{up}:whole"))
    await _work(app)
    await send(app, callback_update(ADMIN_ID, f"ocr:{_rid(telegram)}:go"))
    await _work(app)
    up2 = await _upload(app, telegram, ANA, scan, file_id="F2")
    await send(app, callback_update(ANA, f"book:{up2}:short"))
    await _work(app)
    assert "S0" in telegram.sent("sendMessage")[-1]["text"]
    assert db.ocr_usage(ANA)[0] == 0


async def test_cancel_during_ocr_still_counts(app, telegram, tmp_path, monkeypatch):
    up = await _upload(app, telegram, ANA, _scan(tmp_path, 1))
    await send(app, callback_update(ANA, f"book:{up}:whole"))
    await _work(app)

    def cancelled(*a, **k):
        """The user is removed while the OCR runs."""
        raise proc.ProcCancelled("cancelled")

    monkeypatch.setattr(ocr, "_read_batch", cancelled)
    await send(app, callback_update(ANA, f"ocr:{_rid(telegram)}:go"))
    await _work(app)
    assert telegram.texts()[-1] == bot.CANCELLED and db.ocr_usage(ANA)[0] == 1


async def test_no_ocr_engine(app, telegram, tmp_path, monkeypatch):
    monkeypatch.setattr(ocr, "available", lambda: False)
    up = await _upload(app, telegram, ANA, _scan(tmp_path, 1))
    await send(app, callback_update(ANA, f"book:{up}:whole"))
    await _work(app)
    assert telegram.texts()[-1] == ocr.NO_ENGINE


# ---------- language check ----------

def _fake_sample(monkeypatch, tmp_path, *, script="Latin", iso="en", confidence=95.0):
    """Makes choose_languages see a sample with this script, language and confidence."""
    img = tmp_path / "p.png"
    img.write_bytes(b"x")
    monkeypatch.setattr(ocr, "_render", lambda pdf, pages, workdir, tag: {p: img for p in pages})
    monkeypatch.setattr(ocr, "_script", lambda image, workdir: script)
    row = "5\t1\t1\t1\t1\t1\t0\t0\t10\t10\t{c}\tword"
    monkeypatch.setattr(ocr, "_tesseract", lambda images, langs, workdir, tag, tsv=False:
                        ["\n".join(row.format(c=confidence) for _ in range(50))] * len(images))
    monkeypatch.setattr(ocr, "_detect", lambda text: iso)


def test_unsupported_language_is_refused_with_the_supported_list(monkeypatch, tmp_path):
    _fake_sample(monkeypatch, tmp_path, iso="pl", confidence=40)
    with pytest.raises(ocr.UnsupportedLanguage, match="looks like Polish.*Supported: English"):
        ocr.choose_languages(tmp_path / "s.pdf", [0, 1, 2], tmp_path)


def test_unsupported_script_is_refused(monkeypatch, tmp_path):
    _fake_sample(monkeypatch, tmp_path, script="Cyrillic")
    with pytest.raises(ocr.UnsupportedLanguage, match="Cyrillic script"):
        ocr.choose_languages(tmp_path / "s.pdf", [0], tmp_path)


def test_close_language_with_good_confidence_is_not_refused(monkeypatch, tmp_path):
    db.set_setting("ocr_languages", "eng,slv")
    _fake_sample(monkeypatch, tmp_path, iso="hr", confidence=88)  # Slovenian text that looks Croatian
    assert ocr.choose_languages(tmp_path / "s.pdf", [0], tmp_path) == "eng+slv"


def test_detected_installed_language_is_used_alone(monkeypatch, tmp_path):
    db.set_setting("ocr_languages", "eng,slv")
    _fake_sample(monkeypatch, tmp_path, iso="sl")
    assert ocr.choose_languages(tmp_path / "s.pdf", [0], tmp_path) == "slv"


async def test_unsupported_language_reaches_the_user(app, telegram, tmp_path, monkeypatch):
    def refuse(*a, **k):
        """The sample is in a language that isn't installed."""
        raise ocr.UnsupportedLanguage("⚠️ This scan looks like Polish, which text recognition doesn't support yet. "
                                      "Supported: English.")

    monkeypatch.setattr(ocr, "choose_languages", refuse)
    up = await _upload(app, telegram, ANA, _scan(tmp_path, 1))
    await send(app, callback_update(ANA, f"book:{up}:whole"))
    await _work(app)
    assert telegram.texts()[-1].startswith("⚠️ This scan looks like Polish") and db.ocr_usage(ANA)[0] == 0


# ---------- limits ----------

async def test_limit_commands_for_ocr(app, telegram):
    await send(app, msg_update(ADMIN_ID, "/limit ocr 2"))
    assert telegram.texts()[-1] == "✅ OCR limit for everyone: 2."
    await send(app, msg_update(ADMIN_ID, f"/limit ocr {ANA} 7"))
    assert access.ocr_limit(ANA) == (7, False) and access.ocr_limit(BOB) == (2, True)
    await send(app, msg_update(ANA, "/limit"))
    assert telegram.texts()[-1].split("\n")[1] == "🔍 Today: 0 of your 7 scanned documents (OCR) (last 24 h). 7 left."
    await send(app, msg_update(ADMIN_ID, "/users"))
    assert "OCR 0/7" in "\n".join(telegram.texts())


@pytest.mark.slow
def test_rapidocr_reads_a_page_in_the_sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OCR_ENGINE", "rapidocr")
    scan = make_scan(tmp_path / "s.pdf", [TEXT])
    texts = ocr._read_batch(scan, [0], "eng", tmp_path, "t")
    assert "quick brown fox" in texts[0].lower()
