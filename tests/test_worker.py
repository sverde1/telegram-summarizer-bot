"""The job worker must survive every Telegram failure and keep processing the queue."""
import asyncio

import pytest
from telegram.error import Forbidden, RetryAfter

import bot
from summarizer import db, pipeline

from conftest import ADMIN_ID

USER = 60


def _result(text="A summary.") -> pipeline.Result:
    """A finished summary result."""
    summary = {"title": "T", "is_clickbait": False, "clickbait_answer": "", "summary": text}
    return pipeline.Result("youtube", "id", "https://youtu.be/id", {"title": "T"}, "", "captions", "en", summary)


async def _run(app, jobs: list[bot.Job], outcome) -> None:
    """Queues jobs, runs the worker until they're all handled, then stops it.

    Args:
        app: The test application.
        jobs: Jobs to queue.
        outcome: Function (job) -> Result, or raising, standing in for the pipeline.
    """
    for job in jobs:
        await bot.queue.put(job)
    task = asyncio.create_task(bot.worker(app))
    try:
        await asyncio.wait_for(bot.queue.join(), 5)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not task.done() or task.cancelled()


def _job(n: int, **kw) -> bot.Job:
    """A queued job for USER with its own request row."""
    return bot.Job(f"https://youtu.be/{n:011d}", chat_id=USER, status_id=n, user_id=USER,
                   request_id=db.add_request(USER, "u", "summary"), **kw)


@pytest.fixture
def fake_pipeline(monkeypatch):
    """Replaces pipeline.run: job URLs ending in 1 fail with a PipelineError, others succeed."""
    def run(url, progress, **kw):
        """Fails or succeeds depending on the URL."""
        if url.endswith("1"):
            raise pipeline.PipelineError("Video is unavailable.")
        return _result(f"summary for {url[-2:]}")
    monkeypatch.setattr(pipeline, "run", run)


async def test_user_blocking_the_bot_does_not_stop_the_worker(app, telegram, fake_pipeline):
    telegram.fail["sendMessage"] = Forbidden("bot was blocked by the user")
    first, second = _job(2), _job(4)
    await _run(app, [first, second], None)
    assert db.recent_requests(USER)[1]["error"] == "user blocked the bot"
    assert any("summary for 04" in t for t in telegram.texts())  # the next job still ran


async def test_failed_job_with_unreachable_user_does_not_stop_the_worker(app, telegram, fake_pipeline):
    telegram.fail["editMessageText"] = Forbidden("bot was blocked by the user")
    await _run(app, [_job(1), _job(4)], None)  # job 1 fails, and its error report can't be delivered
    assert any("summary for 04" in t for t in telegram.texts())


async def test_flood_control_is_waited_out(app, telegram, fake_pipeline, monkeypatch):
    telegram.fail["sendMessage"] = RetryAfter(0)
    await _run(app, [_job(4)], None)
    sends = [d for d in telegram.sent("sendMessage") if "summary for 04" in d["text"]]
    assert len(sends) == 2  # refused once, then delivered
    assert db.recent_requests(USER)[0]["status"] == "done"


async def test_deleted_status_message_is_not_an_error(app, telegram, fake_pipeline):
    from telegram.error import BadRequest
    telegram.fail["deleteMessage"] = BadRequest("message to delete not found")
    await _run(app, [_job(4)], None)
    assert db.recent_requests(USER)[0]["status"] == "done"
    assert not any("Unexpected error" in t for t in telegram.texts())
