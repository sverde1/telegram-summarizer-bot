"""One cancel routine for every job state; a cancel wins over whatever error it caused."""
import asyncio

import pytest

import access
from summarizer import db, pipeline

from conftest import callback_update, send
from tgbot import jobs, state, texts

USER = 60


@pytest.fixture(autouse=True)
def approved():
    """The test user is approved."""
    access.set_state(USER, "allowed")


def _job(n=1, **kw):
    """A registered job for USER."""
    job = state.Job(f"https://youtu.be/{'a' * 10}{n}", chat_id=USER, status_id=n, user_id=USER,
                  request_id=db.add_request(USER, "u", "summary"), **kw)
    state.jobs[job.request_id] = job
    state.user_jobs[USER] += 1
    return job


async def test_cancelling_a_queued_job_frees_the_slot_at_once(app, telegram):
    job = _job()
    await state.queue.put(job)
    assert await jobs.cancel_job(app, job, texts.CANCELLED)
    assert state.user_jobs.get(USER, 0) == 0 and job.request_id not in state.jobs
    assert telegram.sent("editMessageText")[-1]["text"].startswith(texts.CANCELLED + "\n")
    assert db.recent_requests(USER)[0]["status"] == "cancelled"
    assert not await jobs.cancel_job(app, job, texts.CANCELLED)  # second cancel: nothing to do


async def test_a_replay_cancelled_before_it_starts_is_cleaned_up(app, telegram):
    job = _job()
    started = []

    async def replay():
        """Would deliver; must never run."""
        started.append(1)

    state.delayed[job.request_id] = asyncio.create_task(replay())
    await jobs.cancel_job(app, job, texts.ACCESS_REMOVED)  # before the task got to run
    await asyncio.sleep(0)
    assert started == [] and state.delayed == {} and state.jobs == {} and state.user_jobs == {}


async def test_cancel_button_on_a_queued_job_is_reported(app, telegram):
    job = _job()
    await send(app, callback_update(USER, f"cancel:{job.request_id}"))
    assert telegram.sent("editMessageText")[-1]["text"].startswith(texts.CANCELLED + "\n")


async def _run_once(app, job, run):
    """Runs one job through the worker with `run` standing in for the pipeline."""
    original = pipeline.run
    pipeline.run = run
    try:
        await state.queue.put(job)
        task = asyncio.create_task(jobs.worker(app))
        await asyncio.wait_for(state.queue.join(), 5)
        task.cancel()
    finally:
        pipeline.run = original


async def test_cancel_landing_on_a_cached_answer_is_reported(app, telegram):
    job = _job()

    def cached(url, progress, **kw):
        """Returns at once from the cache, while the user is removed meanwhile."""
        job.cancel_reason = texts.ACCESS_REMOVED
        return pipeline.Result("youtube", "x", url, {"title": "T"}, "", "captions", "en",
                               {"title": "T", "is_clickbait": False, "clickbait_answer": "", "summary": "S"},
                               cached=True)

    await _run_once(app, job, cached)
    assert telegram.sent("editMessageText")[-1]["text"] == texts.ACCESS_REMOVED
    assert not telegram.sent("sendMessage")  # the summary was not delivered
    assert db.recent_requests(USER)[0]["status"] == "cancelled"


async def test_a_download_killed_by_the_cancel_reports_the_cancel(app, telegram):
    job = _job()

    def killed(url, progress, **kw):
        """yt-dlp dies from the cancel's SIGKILL and the pipeline reports a download error."""
        job.cancel_reason = texts.ACCESS_REMOVED
        raise pipeline.PipelineError("⚠️ Couldn't load this video.", detail="yt-dlp killed")

    await _run_once(app, job, killed)
    assert telegram.sent("editMessageText")[-1]["text"] == texts.ACCESS_REMOVED
    assert db.recent_requests(USER)[0]["status"] == "cancelled"


async def test_the_cancel_message_says_what_was_cancelled(app, telegram):
    job = _job()
    await jobs.cancel_job(app, job, texts.CANCELLED)  # still queued: only the link is known
    assert telegram.sent("editMessageText")[-1]["text"] == f"{texts.CANCELLED}\n{job.url}"
    looked_up = _job(2)
    db.update_request(looked_up.request_id, platform="youtube", video_id="v2")
    db.start_video("youtube", "v2", looked_up.url)
    db.update_video("youtube", "v2", title="Cats at night")
    await jobs.cancel_job(app, looked_up, texts.CANCELLED)
    assert telegram.sent("editMessageText")[-1]["text"] == f"{texts.CANCELLED}\n🎬 Cats at night"
    question = state.Job("Why?", USER, 3, user_id=USER, request_id=db.add_request(USER, "Why?", "ask"),
                         ask_of=1, question="Why?", job_kind=state.JobKind.ASK)
    assert jobs.describe(question) == "💬 Why?"
