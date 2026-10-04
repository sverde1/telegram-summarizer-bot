"""Several workers: light jobs don't wait behind a transcription; heavy steps take turns; cancel is per job."""
import asyncio
import threading

import pytest

import access
from summarizer import cpu, db, pipeline

from conftest import callback_update, msg_update, send
from tgbot import jobs, state

ANA, BOB = 60, 61
LONG, CAPTIONED, LONG2 = "https://youtu.be/aaaaaaaaaaa", "https://youtu.be/bbbbbbbbbbb", "https://youtu.be/ccccccccccc"
SUMMARY = {"title": "T", "is_clickbait": False, "clickbait_answer": "", "summary": "S",
           "_stats": {"steps": [["summary", 1.0]], "total": 1.0, "llm": "Codex (gpt-test)"}}


@pytest.fixture
def pipe(monkeypatch, request):
    """Two users; a pipeline whose "long" videos transcribe in the CPU slot until released."""
    for uid in (ANA, BOB):
        access.set_state(uid, "allowed")
    release = {LONG: threading.Event(), LONG2: threading.Event()}
    started = []

    def run(url, progress, **kw):
        """Long videos hold the CPU slot until their event is set; captioned ones return at once."""
        if url in release:
            progress("🗣 Transcribing…", 3600 if url == LONG else 20)
            with cpu.slot(progress, "transcription", 60):
                started.append(url)
                while not release[url].wait(0.05):
                    from summarizer import proc
                    proc.check_cancelled()
        return pipeline.Result("youtube", url[-11:], url, {"title": "T"}, "", "captions", "en", dict(SUMMARY))

    monkeypatch.setattr(pipeline, "run", run)
    request.addfinalizer(lambda: [e.set() for e in release.values()])  # never leave a fake transcription hanging
    return release, started


async def _until(predicate, timeout=5):
    """Waits until predicate() is true."""
    for _ in range(int(timeout / 0.02)):
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("timed out")


def _status(telegram, uid):
    """The last status text the bot showed to uid."""
    return [d["text"] for m, d in telegram.calls if m in ("sendMessage", "editMessageText")
            and d.get("chat_id") == uid][-1]


async def test_a_captioned_video_is_answered_while_another_transcribes(app, telegram, pipe):
    release, started = pipe
    workers = [asyncio.create_task(jobs.worker(app)) for _ in range(2)]
    try:
        await send(app, msg_update(ANA, LONG))
        await _until(lambda: started == [LONG])
        await send(app, msg_update(BOB, CAPTIONED))
        await _until(lambda: db.recent_requests(BOB)[0]["status"] == "done")
        assert db.recent_requests(ANA)[0]["status"] != "done"  # still transcribing
        release[LONG].set()
        await asyncio.wait_for(state.queue.join(), 5)
    finally:
        for w in workers:
            w.cancel()


async def test_two_transcriptions_take_turns_and_a_cancel_stops_only_the_waiting_one(app, telegram, pipe):
    release, started = pipe
    workers = [asyncio.create_task(jobs.worker(app)) for _ in range(2)]
    try:
        await send(app, msg_update(ANA, LONG))
        await _until(lambda: started == [LONG])
        await send(app, msg_update(BOB, LONG2))
        await _until(lambda: "Waiting for a turn on the transcription engine" in _status(telegram, BOB))
        assert started == [LONG]  # Bob's waits for its turn
        rid = db.recent_requests(BOB)[0]["id"]
        await send(app, callback_update(BOB, f"cancel:{rid}"))
        await _until(lambda: db.recent_requests(BOB)[0]["status"] == "cancelled")
        assert db.recent_requests(ANA)[0]["status"] != "done"  # Ana's goes on
        release[LONG].set()
        await _until(lambda: db.recent_requests(ANA)[0]["status"] == "done")
        assert started == [LONG]
    finally:
        for w in workers:
            w.cancel()


def _markup(telegram, uid):
    """The buttons of the last status edit shown to uid."""
    return str([d for m, d in telegram.calls if m == "editMessageText" and d.get("chat_id") == uid][-1]
               .get("reply_markup"))


async def test_a_long_job_can_be_cancelled_from_its_status(app, telegram, pipe):
    release, started = pipe
    workers = [asyncio.create_task(jobs.worker(app))]
    try:
        await send(app, msg_update(ANA, LONG))
        await _until(lambda: started == [LONG])
        await _until(lambda: "cancel:" in _markup(telegram, ANA))  # an hour to go: the button is there
        rid = db.recent_requests(ANA)[0]["id"]
        await send(app, callback_update(BOB, f"cancel:{rid}"))  # not Bob's
        assert telegram.sent("answerCallbackQuery")[-1]["text"] == "Only the person who sent this link can cancel it."
        await send(app, callback_update(ANA, f"cancel:{rid}"))
        await _until(lambda: db.recent_requests(ANA)[0]["status"] == "cancelled")
        assert _status(telegram, ANA).startswith("✖️ Cancelled.") and "cancel:" not in _markup(telegram, ANA)
    finally:
        for w in workers:
            w.cancel()


async def test_short_steps_get_no_cancel_button(app, telegram, pipe):
    release, started = pipe
    workers = [asyncio.create_task(jobs.worker(app))]
    try:
        await send(app, msg_update(BOB, LONG2))  # a 20 s step
        await _until(lambda: started == [LONG2])
        await _until(lambda: "Transcribing" in _status(telegram, BOB))
        assert "cancel:" not in _markup(telegram, BOB)
    finally:
        release[LONG2].set()
        for w in workers:
            w.cancel()
