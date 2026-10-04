"""📄 Download: the Markdown file per kind of summary, and the button's checks."""
import asyncio

import pytest

import access
from summarizer import db

from conftest import ADMIN_ID, callback_update, send
from tgbot import handlers, markdown

ANA, BOB = 60, 61
URL = "https://www.youtube.com/watch?v=abcdefghijk"


@pytest.fixture(autouse=True)
def users():
    """Two approved users."""
    for uid in (ANA, BOB):
        access.set_state(uid, "allowed")


def _video_summary(uid=ANA, transcript="[0:00] It is 70°F today"):
    """A delivered video summary of uid's; returns its request id."""
    db.start_video("youtube", "abcdefghijk", URL)
    db.update_video("youtube", "abcdefghijk", transcript=transcript, status="done")
    rid = db.add_request(uid, "https://youtu.be/abcdefghijk", "summary")
    db.update_request(rid, platform="youtube", video_id="abcdefghijk", status="done")
    db.save_delivered(rid, "video", "Hot day", "• It reaches 70°F", "codex", "gpt-test")
    return rid


def test_filename_is_safe():
    assert markdown.filename('Cats: "the" truth / part 1?') == "Cats the truth  part 1.md"
    assert markdown.filename("???") == "summary.md" and len(markdown.filename("x" * 200)) == 63


def test_video_file_has_summary_in_the_readers_units_and_the_transcript():
    name, data = markdown.build(_video_summary(), ("metric", "c"))
    text = data.decode()
    assert name == "Hot day.md" and text.startswith(f"# Hot day\n\n{URL}\n\n## Summary\n\n• It reaches 21 °C")
    assert text.endswith("## Transcript\n\n[0:00] It is 70°F today\n")  # the source stays as it was said
    assert db.get_delivered(_video_summary())["text"] == "• It reaches 70°F"  # never stored converted


def test_recording_shows_its_label_and_photo_posts_say_there_is_no_transcript():
    rid = _video_summary(transcript="")
    db.update_request(rid, url="🎤 Voice message (0:42)")
    db.save_delivered(rid, "file", "Battery talk", "Talk", None, None)
    text = markdown.build(rid, ("metric", "c"))[1].decode()
    assert "\n\n🎤 Voice message (0:42)\n\n" in text and "No transcript" in text and URL not in text


def test_book_file_is_the_full_text_without_summaries():
    sha = "s" * 64
    db.save_document(sha, format="pdf", pages=2, status="done",
                     chapters=[{"title": "Cats", "start": 0, "end": 1}, {"title": "Dogs", "start": 1, "end": 2}])
    db.save_pages(sha, {0: "Cats sleep.", 1: "Dogs run."}, "text")
    rid = db.add_request(ANA, "📄 b.pdf", "book")
    db.update_request(rid, platform="document", video_id=sha, status="done")
    db.save_delivered(rid, "document", "Pets", "Author: Ana\n\nA SUMMARY", "codex", "m")
    text = markdown.build(rid, ("metric", "c"))[1].decode()
    assert text == "# Pets\n\n## Cats\n\nCats sleep.\n\n## Dogs\n\nDogs run.\n" and "SUMMARY" not in text


def test_nothing_to_build_when_the_text_is_gone():
    rid = db.add_request(ANA, "📄 b.pdf", "book")
    db.update_request(rid, platform="document", video_id="gone")
    db.save_delivered(rid, "document", "Pets", "x")
    assert markdown.build(rid, ("metric", "c")) is None and markdown.build(9999, ("metric", "c")) is None


async def test_button_sends_the_file_to_the_owner_or_an_admin_once_per_interval(app, telegram, monkeypatch):
    rid = _video_summary()
    await send(app, callback_update(BOB, f"md:{rid}"))
    assert not telegram.sent("sendDocument") and telegram.sent("answerCallbackQuery")[-1]["text"] == \
        "This isn't available."
    await send(app, callback_update(ANA, f"md:{rid}"))
    await send(app, callback_update(ANA, f"md:{rid}"))  # a second tap right away: not sent again
    assert len(telegram.sent("sendDocument")) == 1 and telegram.sent("sendDocument")[0]["chat_id"] == ANA
    assert telegram.sent("answerCallbackQuery")[-1]["text"] == "Already sent."
    await send(app, callback_update(ADMIN_ID, f"md:{rid}"))  # throttled per summary, admins too
    monkeypatch.setattr(handlers, "MD_EVERY", 0)
    await send(app, callback_update(ADMIN_ID, f"md:{rid}"))
    assert len(telegram.sent("sendDocument")) == 2


async def test_button_says_when_too_old_or_too_large(app, telegram, monkeypatch):
    rid = db.add_request(ANA, "📄 b.pdf", "book")
    await send(app, callback_update(ANA, f"md:{rid}"))
    assert "too old" in telegram.sent("answerCallbackQuery")[-1]["text"]
    monkeypatch.setattr(markdown, "MAX_BYTES", 10)
    await send(app, callback_update(ANA, f"md:{_video_summary()}"))
    assert telegram.sent("answerCallbackQuery")[-1]["text"] == "⚠️ Too large to send as a file."
    assert not telegram.sent("sendDocument")
