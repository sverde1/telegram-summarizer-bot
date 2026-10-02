"""🔊 Listen end to end: the button, the voice job, reuse and pacing, limits, failures."""
import asyncio
import re

import pytest
from telegram.error import BadRequest

import access
import bot
from summarizer import db, pipeline, proc, tts

from conftest import ADMIN_ID, callback_update, msg_update, send

ANA, BOB = 60, 61
URL = "https://youtu.be/abcdefghijk"
SUMMARY = {"title": "Cats", "is_clickbait": False, "clickbait_answer": "", "summary": "• They sleep a lot",
           "_stats": {"steps": [["lookup", 1.0], ["summary", 2.0]], "total": 3.0, "llm": "Codex (gpt-test)"}}


@pytest.fixture
def voice(monkeypatch, tmp_path):
    """Voice messages on, a fake Kokoro (writes a small file), and a pipeline returning one summary."""
    for uid in (ANA, BOB):
        access.set_state(uid, "allowed")
    monkeypatch.setattr(tts, "_ready", True)
    made = []

    def synthesize(pieces, workdir, **kw):
        """Pretends to speak; records the text."""
        made.append(" ".join(pieces))
        ogg = workdir / "speech.ogg"
        ogg.write_bytes(b"OggS fake")
        return ogg

    monkeypatch.setattr(tts, "synthesize", synthesize)
    monkeypatch.setattr(pipeline, "run", lambda url, *a, **k: pipeline.Result(
        "youtube", "abcdefghijk", URL, {"title": "Cats"}, "", "captions", "en", dict(SUMMARY)))
    real_sleep = asyncio.sleep
    monkeypatch.setattr(bot.asyncio, "sleep", lambda s: real_sleep(min(s, 0.01)))
    return made


async def _work(app):
    """Runs the worker until the queue and any paced deliveries are done."""
    task = asyncio.create_task(bot.worker(app))
    await asyncio.wait_for(bot.queue.join(), 10)
    for _ in range(300):
        if not bot._delayed:
            break
        await asyncio.sleep(0.01)
    task.cancel()


async def _summary(app, telegram, uid) -> int:
    """Gets a summary for uid; returns its request id from the 🔊 button."""
    await send(app, msg_update(uid, URL))
    await _work(app)
    markup = str(telegram.sent("sendMessage")[-1].get("reply_markup"))
    return int(re.search(r"voice:(\d+)", markup).group(1))


async def test_listen_makes_a_voice_message_and_reuses_it(app, telegram, voice):
    rid = await _summary(app, telegram, ANA)
    await send(app, callback_update(ANA, f"voice:{rid}"))
    await _work(app)
    sent = telegram.sent("sendVoice")[-1]
    assert sent["chat_id"] == ANA and "🔊 Cats" in sent["caption"] and voice == ["Cats. They sleep a lot."]
    key = tts.key("Cats. They sleep a lot.", "en-us", "af_heart")
    assert db.get_voice(key, 86400)["file_id"].startswith("VOICE")
    req = db.recent_requests(ANA)[0]
    assert (req["kind"], req["status"], req["cached"]) == ("voice", "done", 0)
    assert db.usage(ANA, "daily")[0] == 1 and db.usage(ANA, "voice")[0] == 1  # listening isn't a daily request
    await send(app, callback_update(ANA, f"voice:{rid}"))  # again: reused, nothing made
    await _work(app)
    assert len(voice) == 1 and telegram.sent("sendVoice")[-1]["voice"].startswith("VOICE")
    assert db.usage(ANA, "voice")[0] == 1  # a reused one is free


async def test_another_user_gets_the_made_one_paced(app, telegram, voice):
    rid = await _summary(app, telegram, ANA)
    await send(app, callback_update(ANA, f"voice:{rid}"))
    await _work(app)
    rid_bob = await _summary(app, telegram, BOB)
    await send(app, callback_update(BOB, f"voice:{rid_bob}"))
    await _work(app)
    assert len(voice) == 1  # not made again
    shown = [d["text"] for d in telegram.sent("editMessageText") if d["chat_id"] == BOB]
    assert any("Making the voice message" in t for t in shown)  # paced like a fresh run
    assert telegram.sent("sendVoice")[-1]["chat_id"] == BOB


async def test_buttons_only_for_the_owner_and_no_duplicates(app, telegram, voice):
    rid = await _summary(app, telegram, ANA)
    await send(app, callback_update(BOB, f"voice:{rid}"))
    assert telegram.sent("answerCallbackQuery")[-1]["text"] == "This isn't available."
    await send(app, callback_update(ANA, f"voice:{rid}"))
    await send(app, callback_update(ANA, f"voice:{rid}"))  # tapped twice while queued
    assert telegram.sent("answerCallbackQuery")[-1]["text"] == "Already on its way." and bot.queue.qsize() == 1


async def test_no_button_without_voice_messages_or_for_transcripts(app, telegram, voice, monkeypatch):
    monkeypatch.setattr(tts, "_ready", False)
    await send(app, msg_update(ANA, URL))
    await _work(app)
    assert "voice:" not in str(telegram.sent("sendMessage")[-1].get("reply_markup"))
    monkeypatch.setattr(tts, "_ready", True)
    await send(app, msg_update(ANA, f"/transcript {URL}"))
    await _work(app)
    assert all("voice:" not in str(m.get("reply_markup")) for m in telegram.sent("sendDocument"))


async def test_voice_limit_counts_only_new_ones(app, telegram, voice):
    db.set_setting("tts_limit", "1")
    rid = await _summary(app, telegram, ANA)
    await send(app, callback_update(ANA, f"voice:{rid}"))
    await _work(app)
    rid2 = await _summary(app, telegram, ANA)
    await send(app, callback_update(ANA, f"voice:{rid2}"))  # the same text: reused, so allowed
    await _work(app)
    assert len(voice) == 1 and len(telegram.sent("sendVoice")) == 2
    voice_other = dict(SUMMARY, summary="Something new")
    pipeline.run = lambda url, *a, **k: pipeline.Result("youtube", "x", URL, {"title": "N"}, "", "captions",
                                                         "en", voice_other)
    rid3 = await _summary(app, telegram, ANA)
    await send(app, callback_update(ANA, f"voice:{rid3}"))  # a new one: over the limit
    assert "new voice messages today" in telegram.sent("answerCallbackQuery")[-1]["text"]


async def test_a_stale_telegram_id_is_made_again(app, telegram, voice):
    rid = await _summary(app, telegram, ANA)
    await send(app, callback_update(ANA, f"voice:{rid}"))
    await _work(app)
    telegram.fail["sendVoice"] = BadRequest("Wrong file identifier/http url specified")
    await send(app, callback_update(ANA, f"voice:{rid}"))
    await _work(app)
    assert len(voice) == 2 and telegram.sent("sendVoice")[-1]["voice"] != telegram.sent("sendVoice")[-2]["voice"]


async def test_cancel_while_speaking(app, telegram, voice, monkeypatch):
    rid = await _summary(app, telegram, ANA)

    def cancelled(*a, **k):
        """The user is removed while Kokoro runs."""
        raise proc.ProcCancelled("cancelled")

    monkeypatch.setattr(tts, "synthesize", cancelled)
    await send(app, callback_update(ANA, f"voice:{rid}"))
    await _work(app)
    assert telegram.texts()[-1] == bot.CANCELLED and not telegram.sent("sendVoice")


async def test_whole_book_gets_listen_next_to_the_chapter_options(app, telegram, voice, monkeypatch):
    from summarizer import documents
    job = bot.Job("📄 b.pdf", ANA, 5, user_id=ANA, request_id=db.add_request(ANA, "📄 b.pdf", "book"), upload_id=9)
    result = documents.DocResult("book", "b.pdf", {"chapters": [{}, {}]}, book={"title": "Pets", "summary": "Cats."})
    await bot._deliver(app, job, result, 0)
    rows = telegram.sent("sendMessage")[-1]["reply_markup"]
    assert "book:9:short" in str(rows) and f"voice:{job.request_id}" in str(rows)
    assert db.get_spoken(job.request_id)["text"] == "Pets. Cats."


async def test_old_summary_without_stored_text(app, telegram, voice):
    rid = db.add_request(ANA, URL, "summary")
    await send(app, callback_update(ANA, f"voice:{rid}"))
    assert telegram.sent("answerCallbackQuery")[-1]["text"] == bot.VOICE_TOO_OLD
