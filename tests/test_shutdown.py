"""Stopping the bot tells everyone still waiting, stops their work, and leaves nothing behind."""
import asyncio
import sys

import pytest

import access
from summarizer import db, pipeline, proc
import time
from tgbot import jobs, lifecycle, state, texts

USERS = (60, 61, 62, 63)


@pytest.fixture(autouse=True)
def approved():
    """The test users are approved."""
    for uid in USERS:
        access.set_state(uid, "allowed")


def _job(uid, n):
    """A registered job for `uid`."""
    job = state.Job(f"https://youtu.be/{'a' * 10}{n}", chat_id=uid, status_id=n, user_id=uid,
                  request_id=db.add_request(uid, "u", "summary"))
    state.jobs[job.request_id] = job
    state.user_jobs[uid] += 1
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
    await state.queue.put(running)
    await state.queue.put(queued)
    parked.waiting_since, parked.memory_needed = time.monotonic(), 10 ** 15  # set aside, never fits
    state.waiting_for_memory.append(parked)
    state.delayed[replaying.request_id] = asyncio.create_task(asyncio.sleep(60))
    app.bot_data["workers"] = [asyncio.create_task(jobs.worker(app))]
    await asyncio.wait_for(started.wait(), 5)

    await asyncio.wait_for(lifecycle.post_stop(app), 10)

    told = {d["chat_id"] for d in telegram.sent("editMessageText") if d["text"] == texts.STOPPED}
    assert told == set(USERS)
    assert {r["status"] for uid in USERS for r in db.recent_requests(uid)} == {"cancelled"}
    assert state.jobs == {} and state.user_jobs == {} and state.delayed == {} and state.waiting_for_memory == []
    await asyncio.gather(*app.bot_data["workers"], return_exceptions=True)  # let the cancellation finish
    assert app.bot_data["workers"][0].cancelled()


async def test_a_job_that_cannot_stop_in_time_is_reported_anyway(app, telegram, monkeypatch):
    monkeypatch.setattr(lifecycle, "SHUTDOWN_WAIT", 0.3)
    job = _job(60, 1)
    monkeypatch.setattr(state, "running", {job})  # e.g. stuck in an API call that can't be interrupted
    await lifecycle.post_stop(app)
    assert telegram.sent("editMessageText")[-1]["text"] == texts.STOPPED
    assert db.recent_requests(60)[0]["status"] == "cancelled" and state.jobs == {}
