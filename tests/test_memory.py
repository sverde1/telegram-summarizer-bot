"""Memory guard for Whisper: estimate, wait aside while other jobs run, cancel button, wait limit."""
import asyncio

import pytest

import access
import bot
from summarizer import config, db, media, memory, pipeline

from conftest import ADMIN_ID, callback_update, send
from helpers import meta

GB = memory.GB


def test_estimate_counts_the_model_only_when_not_loaded(monkeypatch):
    monkeypatch.setattr(config, "WHISPER_MODEL", "small")
    hour = memory.whisper_needs(3600, model_loaded=True)
    assert hour == 3600 * 3 * 16000 * 4  # ~0.64 GB of audio-derived data per hour
    assert memory.whisper_needs(3600, model_loaded=False) == hour + int(0.6 * GB)


def test_fit_checks(monkeypatch):
    monkeypatch.setattr(memory, "total", lambda: 16 * GB)
    monkeypatch.setattr(memory, "used_by_bot", lambda: 1 * GB)
    monkeypatch.setattr(memory, "available", lambda: 4 * GB)
    monkeypatch.setattr(config, "WHISPER_RAM_FRACTION", 0.5)
    assert memory.can_ever_fit(8 * GB) and not memory.can_ever_fit(9 * GB)
    assert memory.fits_now(3 * GB)
    assert not memory.fits_now(5 * GB)  # within the cap, but not free right now


@pytest.fixture
def needs_whisper(monkeypatch, llm):
    """A 30-minute YouTube video without captions, so it needs Whisper."""
    monkeypatch.setattr(media, "probe", lambda v: meta(duration=1800, subtitles={}))
    monkeypatch.setattr(media, "fetch_captions", lambda *a: None)


def test_pipeline_waits_when_memory_is_short(needs_whisper, monkeypatch):
    monkeypatch.setattr(memory, "fits_now", lambda needed: False)
    with pytest.raises(memory.NeedsMemory) as e:
        pipeline.run("https://youtu.be/abcdefghijk", lambda *a: None)
    assert e.value.needed > 0
    assert db.get_video("youtube", "abcdefghijk")["status"] == "waiting"  # not "failed": it'll be retried


def test_pipeline_refuses_what_can_never_fit(needs_whisper, monkeypatch):
    monkeypatch.setattr(memory, "can_ever_fit", lambda needed: False)
    with pytest.raises(pipeline.PipelineError, match="too long to transcribe on this machine"):
        pipeline.run("https://youtu.be/abcdefghijk", lambda *a: None)


@pytest.fixture
def two_jobs(monkeypatch):
    """Job 1 (user 60) needs memory until `free[0]` is set; job 2 (user 70) runs normally."""
    free = [False]
    monkeypatch.setattr(bot, "MEMORY_RECHECK", 0.05)
    monkeypatch.setattr(memory, "fits_now", lambda needed: free[0])
    ran = []

    def run(url, progress, **kw):
        """Stands in for the pipeline."""
        if url.endswith("1") and not free[0]:
            raise memory.NeedsMemory(2 * GB)
        ran.append(url[-1])
        return pipeline.Result("youtube", url[-11:], url, {"title": "T"}, "", "captions", "en",
                               {"title": "T", "is_clickbait": False, "clickbait_answer": "", "summary": "S"})

    monkeypatch.setattr(pipeline, "run", run)
    for uid in (60, 70):
        access.set_state(uid, "allowed")
    jobs = [bot.Job(f"https://youtu.be/{'a' * 10}{n}", chat_id=uid, status_id=n, user_id=uid,
                    request_id=db.add_request(uid, "u", "summary")) for n, uid in ((1, 60), (2, 70))]
    for job in jobs:
        bot._jobs[job.request_id] = job
        bot._user_jobs[job.user_id] += 1
    return free, ran, jobs


async def _wait_for(condition, seconds=5):
    """Waits until condition() is true."""
    for _ in range(int(seconds / 0.02)):
        if condition():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("condition not reached")


async def test_waiting_job_steps_aside_then_resumes(app, telegram, two_jobs):
    free, ran, jobs = two_jobs
    for job in jobs:
        await bot.queue.put(job)
    task = asyncio.create_task(bot.worker(app))
    await _wait_for(lambda: ran == ["2"])  # job 2 ran while job 1 waits
    waiting = [d for d in telegram.sent("editMessageText") if d["chat_id"] == 60][-1]
    assert "Not enough free memory" in waiting["text"] and "cancel:" in str(waiting["reply_markup"])
    free[0] = True
    await _wait_for(lambda: ran == ["2", "1"])
    task.cancel()
    assert bot._user_jobs == {} and bot._jobs == {}


async def test_cancel_button_only_for_the_owner(app, telegram, two_jobs):
    free, ran, jobs = two_jobs
    await bot.queue.put(jobs[0])
    task = asyncio.create_task(bot.worker(app))
    await _wait_for(lambda: bot._waiting_for_memory)
    await send(app, callback_update(70, f"cancel:{jobs[0].request_id}"))  # someone else
    assert bot._waiting_for_memory and "Only the person" in telegram.sent("answerCallbackQuery")[-1]["text"]
    await send(app, callback_update(60, f"cancel:{jobs[0].request_id}"))
    task.cancel()
    assert not bot._waiting_for_memory and ran == []
    assert db.recent_requests(60)[0]["status"] == "cancelled"
    assert telegram.sent("editMessageText")[-1]["text"] == bot.CANCELLED


async def test_waiting_gives_up_after_the_limit(app, telegram, two_jobs, monkeypatch):
    free, ran, jobs = two_jobs
    monkeypatch.setattr(config, "WHISPER_RAM_WAIT_MIN", 0)
    await bot.queue.put(jobs[0])
    task = asyncio.create_task(bot.worker(app))
    await _wait_for(lambda: db.recent_requests(60)[0]["status"] == "failed")
    task.cancel()
    assert "Still not enough free memory" in telegram.sent("editMessageText")[-1]["text"]
    assert bot._user_jobs.get(60, 0) == 0
