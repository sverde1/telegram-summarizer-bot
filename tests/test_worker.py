"""The job worker must survive every Telegram failure and keep processing the queue."""
import asyncio

import pytest
from telegram.error import Forbidden, RetryAfter

import access
from summarizer import db, pipeline

from conftest import ADMIN_ID
from tgbot import jobs, runners, state

USER = 60


@pytest.fixture(autouse=True)
def approved_user():
    """The test user is an approved user (the worker cancels jobs of users without access)."""
    access.set_state(USER, "allowed", "Ana", None)


def _result(text="A summary.") -> pipeline.Result:
    """A finished summary result."""
    summary = {"title": "T", "is_clickbait": False, "clickbait_answer": "", "summary": text}
    return pipeline.Result("youtube", "id", "https://youtu.be/id", {"title": "T"}, "", "captions", "en", summary)


async def _run(app, queued: list[state.Job], outcome) -> None:
    """Queues the jobs, runs the worker until they're all handled, then stops it.

    Args:
        app: The test application.
        queued: Jobs to queue.
        outcome: Function (job) -> Result, or raising, standing in for the pipeline.
    """
    for job in queued:
        await state.queue.put(job)
    task = asyncio.create_task(jobs.worker(app))
    try:
        await asyncio.wait_for(state.queue.join(), 5)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert not task.done() or task.cancelled()


def _job(n: int, **kw) -> state.Job:
    """A queued job for USER with its own request row."""
    return state.Job(f"https://youtu.be/{n:011d}", chat_id=USER, status_id=n, user_id=USER,
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


def test_every_job_kind_has_a_runner():
    assert set(runners.RUNNERS) == set(state.JobKind)


@pytest.mark.parametrize("fields", [
    {"upload_id": 3},  # a video job naming an upload
    {"voice_of": 4},  # a video job naming a summary to read
    {"job_kind": state.JobKind.DOCUMENT},  # a document job without its upload
    {"job_kind": state.JobKind.MEDIA, "upload_id": 3, "voice_of": 4},
    {"job_kind": state.JobKind.VOICE},
])
def test_a_job_must_have_the_fields_of_its_kind(fields):
    with pytest.raises(ValueError):
        state.Job("u", chat_id=1, status_id=1, **fields)


def test_jobs_of_each_kind_with_their_fields():
    assert state.Job("u", 1, 1).job_kind is state.JobKind.VIDEO
    for kind in (state.JobKind.MEDIA, state.JobKind.DOCUMENT):
        assert state.Job("u", 1, 1, upload_id=3, job_kind=kind).job_kind is kind
    assert state.Job("u", 1, 1, voice_of=4, job_kind=state.JobKind.VOICE).job_kind is state.JobKind.VOICE
