"""Per-user and global queue limits, and the /again cooldown."""
import time

import pytest

import access
from summarizer import config, db, pipeline

from conftest import ADMIN_ID, msg_update, send
from tgbot import jobs, state

FRIEND = 60
LINK = "https://youtu.be/abcdefghijk"


async def test_a_user_may_queue_three_videos(app, telegram):
    access.set_state(FRIEND, "allowed")
    for _ in range(config.MAX_QUEUED_PER_USER + 1):
        await send(app, msg_update(FRIEND, LINK))
    assert state.queue.qsize() == config.MAX_QUEUED_PER_USER
    assert telegram.texts()[-1].startswith("⏳ You already have 3 requests in the queue")


async def test_admins_are_not_limited(app, telegram):
    for _ in range(config.MAX_QUEUED_PER_USER + 2):
        await send(app, msg_update(ADMIN_ID, LINK))
    assert state.queue.qsize() == config.MAX_QUEUED_PER_USER + 2


async def test_full_queue_refuses_new_links(app, telegram):
    access.set_state(FRIEND, "allowed")
    for _ in range(config.MAX_QUEUE):
        await send(app, msg_update(ADMIN_ID, LINK))
    await send(app, msg_update(FRIEND, LINK))
    assert state.queue.qsize() == config.MAX_QUEUE
    assert telegram.texts()[-1] == "⏳ The bot is busy right now. Please try again in a few minutes."


async def test_jobs_waiting_for_memory_count_toward_the_total(app, telegram):
    access.set_state(FRIEND, "allowed")
    for n in range(config.MAX_QUEUE):  # e.g. many users' long videos waiting for memory
        job = state.Job("u", chat_id=1, status_id=n, user_id=1000 + n, request_id=10_000 + n)
        state.jobs[job.request_id] = job
        state.waiting_for_memory.append(job)
    await send(app, msg_update(FRIEND, LINK))
    assert state.queue.qsize() == 0 and "busy" in telegram.texts()[-1]


async def test_finished_jobs_free_the_slot(app, telegram, monkeypatch):
    import asyncio
    access.set_state(FRIEND, "allowed")
    monkeypatch.setattr(pipeline, "run", lambda *a, **k: (_ for _ in ()).throw(pipeline.PipelineError("nope")))
    for _ in range(config.MAX_QUEUED_PER_USER):
        await send(app, msg_update(FRIEND, LINK))
    task = asyncio.create_task(jobs.worker(app))
    await asyncio.wait_for(state.queue.join(), 5)
    task.cancel()
    assert state.user_jobs[FRIEND] == 0
    await send(app, msg_update(FRIEND, LINK))
    assert state.queue.qsize() == 1


def _done_again(user: int, minutes_ago: float) -> None:
    """Records a finished /again of the test video by `user`, `minutes_ago` minutes ago."""
    rid = db.add_request(user, LINK, "again")
    db.update_request(rid, platform="youtube", video_id="abcdefghijk", status="done")
    with db._db() as c:
        c.execute("UPDATE requests SET finished_at=? WHERE id=?", (time.time() - minutes_ago * 60, rid))


def test_again_cooldown(fake_media, llm):
    _done_again(FRIEND, minutes_ago=3)
    rid = db.add_request(FRIEND, LINK, "again")
    with pytest.raises(pipeline.PipelineError, match="You redid this video 3 min ago; you can redo it again in 7 min"):
        pipeline.run(LINK, lambda *a: None, use_cache=False, request_id=rid, again_limit_user=FRIEND)


def test_again_allowed_after_cooldown_and_for_admins(fake_media, llm):
    _done_again(FRIEND, minutes_ago=config.AGAIN_COOLDOWN_MIN + 1)
    assert not pipeline.run(LINK, lambda *a: None, use_cache=False, again_limit_user=FRIEND).cached
    _done_again(ADMIN_ID, minutes_ago=1)
    pipeline.run(LINK, lambda *a: None, use_cache=False, again_limit_user=None)  # admins: no limit
