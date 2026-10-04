"""What a delivered summary leaves behind for 💬 and 📄: the `delivered` row and every message's id."""
import asyncio

import pytest

import access
from summarizer import db, documents, pipeline

from conftest import msg_update, send
from tgbot import delivery, jobs, state

ANA = 60
URL = "https://youtu.be/abcdefghijk"
STATS = {"steps": [["summary", 2.0]], "total": 2.0, "llm": "Codex (gpt-test)", "backend": "codex",
         "model": "gpt-test"}


def _summary(text="• They sleep a lot", **extra):
    """A video summary dict."""
    return {"title": "Cats", "is_clickbait": False, "clickbait_answer": "", "summary": text, "_stats": STATS,
            **extra}


@pytest.fixture(autouse=True)
def ana():
    """An approved user."""
    access.set_state(ANA, "allowed")


async def _run(app, monkeypatch, result):
    """Sends URL as ANA with the pipeline returning `result`, and runs the worker until it's delivered."""
    monkeypatch.setattr(pipeline, "run", lambda *a, **k: result)
    await send(app, msg_update(ANA, URL))
    task = asyncio.create_task(jobs.worker(app))
    await asyncio.wait_for(state.queue.join(), 5)
    task.cancel()
    return db.recent_requests(ANA)[0]["id"]


async def test_a_summary_is_recorded_with_every_message_id(app, telegram, monkeypatch):
    long = "\n".join(f"• point {i} " + "x" * 300 for i in range(40))  # several messages
    result = pipeline.Result("youtube", "abcdefghijk", URL, {"title": "Cats"}, "", "captions", "en",
                             _summary(long, is_clickbait=True, clickbait_answer="No, only naps."))
    rid = await _run(app, monkeypatch, result)
    row = db.get_delivered(rid)
    assert (row["kind"], row["title"], row["backend"], row["model"]) == ("video", "Cats", "codex", "gpt-test")
    assert row["text"].startswith("Clickbait answer: No, only naps.\n\n• point 0")
    sent = telegram.sent("sendMessage")[1:]  # after the queued notice
    assert len(sent) > 1  # the fake numbers sent messages from 1000: each part maps to the request
    assert sum(1 for m in range(1000, 1100) if db.message_request(ANA, m) == rid) == len(sent)


async def test_transcript_only_results_record_nothing(app, telegram, monkeypatch):
    monkeypatch.setattr(pipeline, "run", lambda *a, **k: pipeline.Result(
        "youtube", "abcdefghijk", URL, {"title": "Cats"}, "[0:00] hi", "captions", "en", None))
    await send(app, msg_update(ANA, f"/transcript {URL}"))
    task = asyncio.create_task(jobs.worker(app))
    await asyncio.wait_for(state.queue.join(), 5)
    task.cancel()
    assert db.get_delivered(db.recent_requests(ANA)[0]["id"]) is None


def test_a_recording_is_titled_by_its_summary_or_label_never_the_shared_row():
    job = state.Job("🎤 Voice message (0:42)", 1, 1)
    r = pipeline.Result("file", "sha", "🎤 Voice message (0:42)", {"title": "Audio file"}, "t", "whisper", "en",
                        {"summary": "Talk", "is_clickbait": True, "clickbait_answer": "x"})
    assert delivery._delivered(job, r)[:3] == ("file", "🎤 Voice message (0:42)", "Talk")
    r.summary["title"] = "Battery talk"
    assert delivery._delivered(job, r)[1] == "Battery talk"


def test_documents_record_the_book_or_its_chapters():
    job = state.Job("📄 b.pdf", 1, 1, upload_id=3, job_kind=state.JobKind.DOCUMENT)
    book = documents.DocResult("book", "b.pdf", {}, book={"title": "Pets", "author": "Ana", "summary": "Cats."},
                               backend="codex", model="gpt-test")
    assert delivery._delivered(job, book) == ("document", "Pets", "Author: Ana\n\nCats.", "codex", "gpt-test")
    each = documents.DocResult("each", "b.pdf", {}, chapters=[(0, "Cats", "Sleep"), (1, "Dogs", "Run")])
    assert delivery._delivered(job, each)[1:3] == ("b.pdf", "Cats\nSleep\n\nDogs\nRun")


def test_ask_history_is_per_user_and_newest_three():
    parent = db.add_request(ANA, URL, "summary")
    for i in range(5):
        rid = db.add_request(ANA, f"q{i}", "ask")
        db.save_delivered(rid, "ask", f"q{i}", f"a{i}", parent_id=parent)
    admin_q = db.add_request(1, "admin q", "ask")
    db.save_delivered(admin_q, "ask", "admin q", "admin a", parent_id=parent)
    assert [h["title"] for h in db.ask_history(parent, ANA)] == ["q2", "q3", "q4"]
