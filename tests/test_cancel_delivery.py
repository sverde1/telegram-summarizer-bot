"""Nothing is delivered after a cancel: not after a flood-control wait, not the next part of a summary."""
import asyncio

import pytest
from telegram.error import RetryAfter

import access
from summarizer import db, pipeline
from tgbot import jobs, state, texts

USER = 60


@pytest.fixture(autouse=True)
def approved():
    """The test user is approved."""
    access.set_state(USER, "allowed")


def _result(text="S") -> pipeline.Result:
    """A fresh (not cached) summary."""
    return pipeline.Result("youtube", "x", "https://youtu.be/x", {"title": "T"}, "", "captions", "en",
                           {"title": "T", "is_clickbait": False, "clickbait_answer": "", "summary": text})


async def _run(app, result):
    """Runs one job for USER through the worker, the pipeline returning `result`."""
    job = state.Job("https://youtu.be/aaaaaaaaaaa", chat_id=USER, status_id=1, user_id=USER,
                  request_id=db.add_request(USER, "u", "summary"))
    state.jobs[job.request_id] = job
    state.user_jobs[USER] += 1
    original = pipeline.run
    pipeline.run = lambda *a, **k: result
    try:
        await state.queue.put(job)
        task = asyncio.create_task(jobs.worker(app))
        await asyncio.wait_for(state.queue.join(), 5)
        task.cancel()
    finally:
        pipeline.run = original
    return job


async def test_removal_during_flood_control_wait_stops_delivery(app, telegram, monkeypatch):
    telegram.fail["sendMessage"] = RetryAfter(1)
    real_sleep = asyncio.sleep

    async def sleep_and_remove(seconds):
        """While waiting out flood control, the admin removes the user."""
        await jobs.cancel_user_jobs(app, USER, texts.ACCESS_REMOVED)
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", sleep_and_remove)
    await _run(app, _result())
    assert len(telegram.sent("sendMessage")) == 1  # only the refused first attempt
    assert telegram.sent("editMessageText")[-1]["text"] == texts.ACCESS_REMOVED
    assert db.recent_requests(USER)[0]["status"] == "cancelled"


async def test_removal_between_parts_of_a_long_summary(app, telegram):
    real_post = telegram.post
    sent = []

    async def post(endpoint, data, **kw):
        """After the first part is delivered, the user is removed."""
        out = await real_post(endpoint, data, **kw)
        if endpoint == "sendMessage":
            sent.append(1)
            if len(sent) == 1:
                await jobs.cancel_user_jobs(app, USER, texts.ACCESS_REMOVED)
        return out

    telegram.post = post
    await _run(app, _result("• a point that goes on and on\n" * 400))  # several messages long
    assert len(telegram.sent("sendMessage")) == 1  # the rest was not sent
    assert db.recent_requests(USER)[0]["status"] == "cancelled"


async def test_fully_delivered_summary_stays_done(app, telegram):
    await _run(app, _result())
    assert db.recent_requests(USER)[0]["status"] == "done"
