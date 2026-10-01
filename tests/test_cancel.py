"""Removing or blocking a user cancels their queued and running jobs at once."""
import asyncio
import sys
import time

import pytest

import access
import bot
from summarizer import db, pipeline, proc

from conftest import ADMIN_ID, callback_update, msg_update, send

FRIEND = 60
LINK = "https://youtu.be/abcdefghijk"


@pytest.fixture(autouse=True)
def friend():
    """An approved user."""
    access.set_state(FRIEND, "allowed", "Ana", None)


async def test_removing_a_user_cancels_their_queued_jobs(app, telegram, monkeypatch):
    ran = []
    monkeypatch.setattr(pipeline, "run", lambda *a, **k: ran.append(1))
    await send(app, msg_update(FRIEND, LINK))
    await send(app, msg_update(FRIEND, "https://youtu.be/bbbbbbbbbbb"))
    await send(app, callback_update(ADMIN_ID, f"remove:{FRIEND}"))
    cancelled = [d for d in telegram.sent("editMessageText") if d["text"] == bot.ACCESS_REMOVED]
    assert len(cancelled) == 2
    assert {r["status"] for r in db.recent_requests(FRIEND)} == {"cancelled"}
    task = asyncio.create_task(bot.worker(app))
    await asyncio.wait_for(bot.queue.join(), 5)
    task.cancel()
    assert ran == [] and bot._user_jobs[FRIEND] == 0 and bot._jobs == {}


async def test_a_running_job_is_killed_and_the_worker_moves_on(app, telegram, monkeypatch):
    started = asyncio.Event()
    loop = asyncio.get_running_loop()

    def slow_pipeline(url, progress, **kw):
        """A pipeline step stuck in a long-running program (like a big download)."""
        done = pipeline.Result("youtube", "c", url, {"title": "T"}, "", "captions", "en", None)
        if url.endswith("ccc"):
            return done  # the next job: finishes at once
        loop.call_soon_threadsafe(started.set)
        proc.run([sys.executable, "-c", "import time; time.sleep(60)"], timeout=120)
        return done

    monkeypatch.setattr(pipeline, "run", slow_pipeline)
    monkeypatch.setattr(bot, "_deliver", lambda *a, **k: asyncio.sleep(0))
    access.set_state(70, "allowed")
    await send(app, msg_update(FRIEND, LINK))
    await send(app, msg_update(70, "https://youtu.be/ccccccccccc"))
    task = asyncio.create_task(bot.worker(app))
    await asyncio.wait_for(started.wait(), 5)
    t0 = time.monotonic()
    await bot.cancel_user_jobs(app, FRIEND, bot.ACCESS_REMOVED)
    await asyncio.wait_for(bot.queue.join(), 5)
    task.cancel()
    assert time.monotonic() - t0 < 3  # killed, not waited out
    assert db.recent_requests(FRIEND)[0]["status"] == "cancelled"
    assert db.recent_requests(70)[0]["status"] == "done"  # the next user's job still ran
    assert any(d["text"] == bot.ACCESS_REMOVED for d in telegram.sent("editMessageText"))


def test_stage_changes_are_checkpoints(monkeypatch):
    st = pipeline.Status(lambda *a: None)
    st.show("one")
    proc.current_job_cancel.set()
    with pytest.raises(proc.ProcCancelled):
        st.show("two")


def test_a_cancelled_video_is_not_marked_failed(monkeypatch, llm):
    from summarizer import media
    from helpers import meta

    def probe(v):
        """Probes, then the job gets cancelled."""
        proc.current_job_cancel.set()
        return meta()

    monkeypatch.setattr(media, "probe", probe)
    with pytest.raises(proc.ProcCancelled):
        pipeline.run(LINK, lambda *a: None)
    assert db.get_video("youtube", "abcdefghijk")["status"] == "cancelled"
