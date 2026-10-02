"""Stopping the bot tells everyone still waiting, stops their work, and leaves nothing behind."""
import asyncio
import sys

import pytest

import access
import bot
from summarizer import db, pipeline, proc

USERS = (60, 61, 62, 63)


@pytest.fixture(autouse=True)
def approved():
    """The test users are approved."""
    for uid in USERS:
        access.set_state(uid, "allowed")


def _job(uid, n):
    """A registered job for `uid`."""
    job = bot.Job(f"https://youtu.be/{'a' * 10}{n}", chat_id=uid, status_id=n, user_id=uid,
                  request_id=db.add_request(uid, "u", "summary"))
    bot._jobs[job.request_id] = job
    bot._user_jobs[uid] += 1
    return job


async def test_every_waiting_user_is_told_and_nothing_is_left(app, telegram, monkeypatch):
    started = asyncio.Event()
    loop = asyncio.get_running_loop()

    def long_download(url, progress, **kw):
        """The running job is in a long download when the bot stops."""
        loop.call_soon_threadsafe(started.set)
        proc.run([sys.executable, "-c", "import time; time.sleep(60)"], timeout=120)

    monkeypatch.setattr(pipeline, "run", long_download)
    running, queued, parked, replaying = (_job(uid, n) for n, uid in enumerate(USERS, 1))
    await bot.queue.put(running)
    await bot.queue.put(queued)
    parked.waiting_since, parked.memory_needed = bot.time.monotonic(), 10 ** 15  # set aside, never fits
    bot._waiting_for_memory.append(parked)
    bot._delayed[replaying.request_id] = asyncio.create_task(asyncio.sleep(60))
    app.bot_data["worker"] = asyncio.create_task(bot.worker(app))
    await asyncio.wait_for(started.wait(), 5)

    await asyncio.wait_for(bot.post_stop(app), 10)

    told = {d["chat_id"] for d in telegram.sent("editMessageText") if d["text"] == bot.STOPPED}
    assert told == set(USERS)
    assert {r["status"] for uid in USERS for r in db.recent_requests(uid)} == {"cancelled"}
    assert bot._jobs == {} and bot._user_jobs == {} and bot._delayed == {} and bot._waiting_for_memory == []
    await asyncio.gather(app.bot_data["worker"], return_exceptions=True)  # let the cancellation finish
    assert app.bot_data["worker"].cancelled()


async def test_a_job_that_cannot_stop_in_time_is_reported_anyway(app, telegram, monkeypatch):
    monkeypatch.setattr(bot, "SHUTDOWN_WAIT", 0.3)
    job = _job(60, 1)
    monkeypatch.setattr(bot, "_running", job)  # e.g. stuck in an API call that can't be interrupted
    await bot.post_stop(app)
    assert telegram.sent("editMessageText")[-1]["text"] == bot.STOPPED
    assert db.recent_requests(60)[0]["status"] == "cancelled" and bot._jobs == {}
