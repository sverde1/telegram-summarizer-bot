"""Uploaded books and documents end to end: intake checks, buttons, the worker, rendering, privacy."""
import asyncio
import re

import pytest
from telegram.error import BadRequest

import access
import bot
from summarizer import books, db, documents, summarize

from conftest import ADMIN_ID, callback_update, doc_update, send
from docs import make_epub, make_pdf, make_scan
from helpers import BookAI

ANA, BOB = 60, 61


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    """Two approved users and a fake AI that answers book questions."""
    for uid in (ANA, BOB):
        access.set_state(uid, "allowed")
    BookAI.calls = []
    monkeypatch.setattr(summarize, "conversation", lambda backend=None, model=None: BookAI())


async def _upload(app, telegram, uid, path, file_id="F1", name=None) -> int:
    """Sends a file as `uid`; returns the upload id from the buttons the bot answered with."""
    telegram.files[file_id] = path
    await send(app, doc_update(uid, name or path.name, path.stat().st_size, file_id=file_id))
    markup = telegram.sent("sendMessage")[-1]["reply_markup"]
    return int(re.search(r"book:(\d+):", str(markup)).group(1))


async def _work(app):
    """Runs the worker until the queue and any replays are done."""
    task = asyncio.create_task(bot.worker(app))
    await asyncio.wait_for(bot.queue.join(), 20)
    for _ in range(300):
        if not bot._delayed:
            break
        await asyncio.sleep(0.02)
    task.cancel()


def _book_pdf(tmp_path, name="book.pdf"):
    """A 4-page PDF with two bookmarked chapters."""
    pages = ["Intro text about cats and their habits", "more about cats and how they sleep all day",
             "Second part on dogs and their loyalty", "the end of the story about dogs and walks"]
    return make_pdf(tmp_path / name, pages, [("Cats", 0), ("Dogs", 2)], title="Pets")


# ---------- intake ----------

async def test_upload_offers_the_choice(app, telegram, tmp_path):
    await _upload(app, telegram, ANA, _book_pdf(tmp_path))
    msg = telegram.sent("sendMessage")[-1]
    assert msg["text"].startswith("📄 book.pdf (") and "How should I summarize it?" in msg["text"]
    assert "Whole book" in str(msg["reply_markup"]) and "By chapter" in str(msg["reply_markup"])


async def test_captioned_document_is_not_read_as_a_link(app, telegram, tmp_path):
    telegram.files["F1"] = _book_pdf(tmp_path)
    await send(app, doc_update(ANA, "book.pdf", 1000, caption="look at this"))
    assert "How should I summarize it?" in telegram.texts()[-1]


@pytest.mark.parametrize("name,size,reply", [
    ("old.doc", 1000, "Convert it to PDF or EPUB"),
    ("photo.zip", 1000, "I can summarize PDF, EPUB, DOCX or TXT files"),
    ("huge.pdf", 21 * 1024 ** 2, "larger than 20 MB"),
])
async def test_unusable_uploads_are_refused(app, telegram, name, size, reply):
    await send(app, doc_update(ANA, name, size))
    assert reply in telegram.texts()[-1] and db.get_upload(1) is None


async def test_strangers_cannot_upload(app, telegram):
    await send(app, doc_update(99, "book.pdf", 1000))
    assert db.get_upload(1) is None


async def test_buttons_of_someone_elses_upload_are_refused(app, telegram, tmp_path):
    up = await _upload(app, telegram, ANA, _book_pdf(tmp_path))
    await send(app, callback_update(BOB, f"book:{up}:whole"))
    assert telegram.sent("answerCallbackQuery")[-1]["text"] == "This isn't available."
    assert bot.queue.qsize() == 0


# ---------- summaries ----------

async def test_whole_book(app, telegram, tmp_path):
    up = await _upload(app, telegram, ANA, _book_pdf(tmp_path))
    await send(app, callback_update(ANA, f"book:{up}:whole"))
    await _work(app)
    text = telegram.sent("sendMessage")[-1]["text"]
    assert "<b>Title:</b>\nBook T" in text and "<b>Author:</b>\nAna" in text and "Whole book." in text
    assert "PDF · 4 pages" in text and "reading the file" in text
    req = db.recent_requests(ANA)[0]
    assert (req["status"], req["kind"], req["url"]) == ("done", "book", "📄 book.pdf")
    assert telegram.sent("deleteMessage")  # the status message


async def test_whole_book_offers_the_chapters_too(app, telegram, tmp_path):
    up = await _upload(app, telegram, ANA, _book_pdf(tmp_path))
    await send(app, callback_update(ANA, f"book:{up}:whole"))
    await _work(app)
    markup = str(telegram.sent("sendMessage")[-1]["reply_markup"])
    assert f"book:{up}:short" in markup and f"book:{up}:each" in markup and f"book:{up}:pick" in markup
    assert f"book:{up}:back" not in markup
    await send(app, callback_update(ANA, f"book:{up}:short"))  # the button works like the menu's
    await _work(app)
    assert "<b>Cats</b>\nS0" in telegram.sent("sendMessage")[-1]["text"]


async def test_single_chapter_document_gets_no_chapter_button(app, telegram, tmp_path):
    (tmp_path / "memo.txt").write_text("A short memo about the budget for next year. " * 20)
    up = await _upload(app, telegram, ANA, tmp_path / "memo.txt")
    await send(app, callback_update(ANA, f"book:{up}:whole"))
    await _work(app)
    assert telegram.sent("sendMessage")[-1].get("reply_markup") is None


async def test_by_chapter_menu_and_short_mode(app, telegram, tmp_path):
    up = await _upload(app, telegram, ANA, make_epub(tmp_path / "b.epub", [("One", "a " * 50), ("Two", "b " * 50)]))
    await send(app, callback_update(ANA, f"book:{up}:chapters"))
    assert "Pick a chapter" in str(telegram.sent("editMessageReplyMarkup")[-1])
    await send(app, callback_update(ANA, f"book:{up}:short"))
    await _work(app)
    text = telegram.sent("sendMessage")[-1]["text"]
    assert "<b>One</b>\nS0" in text and "<b>Two</b>\nS1" in text and "EPUB" in text


async def test_each_mode_sends_one_message_per_chapter(app, telegram, tmp_path):
    up = await _upload(app, telegram, ANA, _book_pdf(tmp_path))
    await send(app, callback_update(ANA, f"book:{up}:each"))
    await _work(app)
    sent = [m["text"] for m in telegram.sent("sendMessage")[-2:]]
    assert sent[0].startswith("<b>1/2</b> <b>Cats</b>") and sent[1].startswith("<b>2/2</b> <b>Dogs</b>")
    assert "<i>" not in sent[0] and "<i>" in sent[1]  # the footer only once, at the end


async def test_pick_a_chapter_counts_once(app, telegram, tmp_path):
    up = await _upload(app, telegram, ANA, _book_pdf(tmp_path))
    await send(app, callback_update(ANA, f"book:{up}:pick"))
    await _work(app)
    listing = telegram.sent("editMessageText")[-1]
    assert "pick a chapter" in listing["text"] and "book:%d:ch:1" % up in str(listing["reply_markup"])
    assert db.recent_requests(ANA)[0]["status"] == "waiting" and not telegram.sent("deleteMessage")
    await send(app, callback_update(ANA, f"book:{up}:ch:1"))
    await _work(app)
    assert "<b>Dogs</b>\nS1" in telegram.sent("sendMessage")[-1]["text"]
    assert db.usage(ANA, "daily")[0] == 1 and db.recent_requests(ANA)[0]["status"] == "done"
    await send(app, callback_update(ANA, f"book:{up}:ch:0"))  # another chapter: a new request
    await _work(app)
    assert db.usage(ANA, "daily")[0] == 2


async def test_pick_is_immediate_once_the_upload_was_read(app, telegram, tmp_path):
    up = await _upload(app, telegram, ANA, _book_pdf(tmp_path))
    await send(app, callback_update(ANA, f"book:{up}:whole"))
    await _work(app)
    requests = len(db.recent_requests(ANA))
    await send(app, callback_update(ANA, f"book:{up}:pick"))  # no worker running: it must not need one
    listing = telegram.sent("sendMessage")[-1]
    assert "pick a chapter" in listing["text"] and f"book:{up}:ch:1" in str(listing["reply_markup"])
    assert bot.queue.qsize() == 0 and len(db.recent_requests(ANA)) == requests  # nothing queued or counted
    await send(app, callback_update(ANA, f"book:{up}:ch:1"))
    await _work(app)
    assert "<b>Dogs</b>\nS1" in telegram.sent("sendMessage")[-1]["text"]
    assert len(db.recent_requests(ANA)) == requests + 1  # the chapter is the request


async def test_pick_on_someone_elses_copy_still_reads_it_first(app, telegram, tmp_path):
    pdf = _book_pdf(tmp_path)
    up = await _upload(app, telegram, ANA, pdf)
    await send(app, callback_update(ANA, f"book:{up}:whole"))
    await _work(app)
    up2 = await _upload(app, telegram, BOB, pdf, file_id="F2")
    await send(app, callback_update(BOB, f"book:{up2}:pick"))
    assert bot.queue.qsize() == 1  # a job, like a fresh upload: nothing reveals the file was known


async def test_chapter_list_pages(app, telegram, tmp_path):
    chapters = [(f"Chapter {i}", "text " * 20) for i in range(1, 13)]
    up = await _upload(app, telegram, ANA, make_epub(tmp_path / "b.epub", chapters))
    await send(app, callback_update(ANA, f"book:{up}:pick"))
    await _work(app)
    assert "(page 1 of 2)" in telegram.sent("editMessageText")[-1]["text"]
    await send(app, callback_update(ANA, f"book:{up}:pg:1"))
    page2 = telegram.sent("editMessageText")[-1]
    assert "(page 2 of 2)" in page2["text"] and "12. Chapter 12" in str(page2["reply_markup"])


async def test_short_mode_fits_three_messages_for_thirty_chapters():
    doc = {"format": "pdf", "pages": 300, "chapters": []}
    summary = "w " * (books.short_budget(30) // 2)
    r = documents.DocResult("short", "big.pdf", doc, chapters=[(i, f"Chapter {i}", summary) for i in range(30)],
                            llm="Codex (x)")
    assert len(bot.render_document(r)) <= 3


# ---------- failures ----------

async def test_empty_document(app, telegram, tmp_path):
    (tmp_path / "empty.txt").write_text("   \n\n  ")
    up = await _upload(app, telegram, ANA, tmp_path / "empty.txt")
    await send(app, callback_update(ANA, f"book:{up}:whole"))
    await _work(app)
    assert telegram.texts()[-1] == documents.EMPTY and BookAI.calls == []


async def test_download_failure(app, telegram, tmp_path):
    up = await _upload(app, telegram, ANA, _book_pdf(tmp_path))
    telegram.fail["getFile"] = BadRequest("File is too big")
    await send(app, callback_update(ANA, f"book:{up}:whole"))
    await _work(app)
    assert telegram.texts()[-1] == bot.TOO_BIG


# ---------- cache and privacy ----------

async def test_second_request_reuses_the_text_and_summary(app, telegram, tmp_path):
    up = await _upload(app, telegram, ANA, _book_pdf(tmp_path))
    await send(app, callback_update(ANA, f"book:{up}:whole"))
    await _work(app)
    await send(app, callback_update(ANA, f"book:{up}:whole"))
    await _work(app)
    assert len(BookAI.calls) == 1 and "from cache" in telegram.sent("sendMessage")[-1]["text"]
    assert len(telegram.sent("getFile")) == 1  # not downloaded again


async def test_same_file_from_someone_else_is_replayed_like_a_fresh_run(app, telegram, tmp_path, monkeypatch):
    real_sleep = asyncio.sleep
    monkeypatch.setattr(bot.asyncio, "sleep", lambda s: real_sleep(s / 1000))
    pdf = _book_pdf(tmp_path)
    up = await _upload(app, telegram, ADMIN_ID, pdf)
    await send(app, callback_update(ADMIN_ID, f"book:{up}:whole"))
    await _work(app)
    up2 = await _upload(app, telegram, BOB, pdf, file_id="F2", name="mine.pdf")
    await send(app, callback_update(BOB, f"book:{up2}:whole"))
    await _work(app)
    text = telegram.sent("sendMessage")[-1]["text"]
    assert "cache" not in text and "reading the file" in text and "Whole book." in text
    stages = [d["text"] for d in telegram.sent("editMessageText") if d["chat_id"] == BOB]
    assert any("Reading the file" in s for s in stages)
    assert len(BookAI.calls) == 1


async def test_pick_on_a_known_file_shows_the_list_after_the_pacing(app, telegram, tmp_path, monkeypatch):
    real_sleep = asyncio.sleep
    monkeypatch.setattr(bot.asyncio, "sleep", lambda s: real_sleep(s / 1000))
    pdf = _book_pdf(tmp_path)
    up = await _upload(app, telegram, ADMIN_ID, pdf)
    await send(app, callback_update(ADMIN_ID, f"book:{up}:whole"))
    await _work(app)
    up2 = await _upload(app, telegram, BOB, pdf, file_id="F2", name="mine.pdf")
    await send(app, callback_update(BOB, f"book:{up2}:pick"))
    await _work(app)
    shown = [d["text"] for d in telegram.sent("editMessageText") if d["chat_id"] == BOB]
    assert any("Listing the chapters" in t for t in shown) and "pick a chapter" in shown[-1]
    assert db.recent_requests(BOB)[0]["status"] == "waiting"
