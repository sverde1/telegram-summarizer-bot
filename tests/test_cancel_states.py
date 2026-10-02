"""One cancel routine for every job state; a cancel wins over whatever error it caused."""
import asyncio

import pytest

import access
import bot
from summarizer import db, pipeline

from conftest import callback_update, send

USER = 60


@pytest.fixture(autouse=True)
def approved():
    """The test user is approved."""
    access.set_state(USER, "allowed")


def _job(n=1, **kw):
    """A registered job for USER."""
    job = bot.Job(f"https://youtu.be/{'a' * 10}{n}", chat_id=USER, status_id=n, user_id=USER,
                  request_id=db.add_request(USER, "u", "summary"), **kw)
    bot._jobs[job.request_id] = job
    bot._user_jobs[USER] += 1
    return job


async def test_cancelling_a_queued_job_frees_the_slot_at_once(app, telegram):
    job = _job()
    await bot.queue.put(job)
    assert await bot._cancel_job(app, job, bot.CANCELLED)
    assert bot._user_jobs.get(USER, 0) == 0 and job.request_id not in bot._jobs
    assert telegram.sent("editMessageText")[-1]["text"] == bot.CANCELLED
    assert db.recent_requests(USER)[0]["status"] == "cancelled"
    assert not await bot._cancel_job(app, job, bot.CANCELLED)  # second cancel: nothing to do


async def test_a_replay_cancelled_before_it_starts_is_cleaned_up(app, telegram):
    job = _job()
    started = []

    async def replay():
        """Would deliver; must never run."""
        started.append(1)

    bot._delayed[job.request_id] = asyncio.create_task(replay())
    await bot._cancel_job(app, job, bot.ACCESS_REMOVED)  # before the task got to run
    await asyncio.sleep(0)
    assert started == [] and bot._delayed == {} and bot._jobs == {} and bot._user_jobs == {}


async def test_cancel_button_on_a_queued_job_is_reported(app, telegram):
    job = _job()
    await send(app, callback_update(USER, f"cancel:{job.request_id}"))
    assert telegram.sent("editMessageText")[-1]["text"] == bot.CANCELLED


async def _run_once(app, job, run):
    """Runs one job through the worker with `run` standing in for the pipeline."""
    original = pipeline.run
    pipeline.run = run
    try:
        await bot.queue.put(job)
        task = asyncio.create_task(bot.worker(app))
        await asyncio.wait_for(bot.queue.join(), 5)
        task.cancel()
    finally:
        pipeline.run = original


async def test_cancel_landing_on_a_cached_answer_is_reported(app, telegram):
    job = _job()

    def cached(url, progress, **kw):
        """Returns at once from the cache, while the user is removed meanwhile."""
        job.cancel_reason = bot.ACCESS_REMOVED
        return pipeline.Result("youtube", "x", url, {"title": "T"}, "", "captions", "en",
                               {"title": "T", "is_clickbait": False, "clickbait_answer": "", "summary": "S"},
                               cached=True)

    await _run_once(app, job, cached)
    assert telegram.sent("editMessageText")[-1]["text"] == bot.ACCESS_REMOVED
    assert not telegram.sent("sendMessage")  # the summary was not delivered
    assert db.recent_requests(USER)[0]["status"] == "cancelled"


async def test_a_download_killed_by_the_cancel_reports_the_cancel(app, telegram):
    job = _job()

    def killed(url, progress, **kw):
        """yt-dlp dies from the cancel's SIGKILL and the pipeline reports a download error."""
        job.cancel_reason = bot.ACCESS_REMOVED
        raise pipeline.PipelineError("⚠️ Couldn't load this video.", detail="yt-dlp killed")

    await _run_once(app, job, killed)
    assert telegram.sent("editMessageText")[-1]["text"] == bot.ACCESS_REMOVED
    assert db.recent_requests(USER)[0]["status"] == "cancelled"
