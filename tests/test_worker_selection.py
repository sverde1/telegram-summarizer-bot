"""The worker survives failures while picking the next job (not only while running one)."""
import asyncio
import sqlite3

import pytest

import access
from summarizer import config, db, memory, pipeline
import time
from tgbot import jobs, state, texts

USER_A, USER_B, USER_C = 60, 70, 80


def _parked(uid, n, waiting_since):
    """A registered job set aside for memory since `waiting_since`."""
    job = state.Job(f"https://youtu.be/{'a' * 10}{n}", chat_id=uid, status_id=n, user_id=uid,
                  request_id=db.add_request(uid, "u", "summary"), waiting_since=waiting_since, memory_needed=1)
    state.jobs[job.request_id] = job
    state.user_jobs[uid] += 1
    state.waiting_for_memory.append(job)
    return job


async def test_cancel_during_an_expiry_report_does_not_kill_the_worker(app, telegram, monkeypatch):
    for uid in (USER_A, USER_B, USER_C):
        access.set_state(uid, "allowed")
    monkeypatch.setattr(memory, "fits_now", lambda needed: False)
    monkeypatch.setattr(config, "WHISPER_RAM_WAIT_MIN", 1)
    expired = _parked(USER_A, 1, waiting_since=-10_000)  # long past its wait limit
    _parked(USER_B, 2, waiting_since=time.monotonic())
    real_fail = jobs.fail

    async def fail_and_cancel_b(app_, job, msg, detail=None):
        """While A's expiry is reported, B's user is removed (the reproduced interleaving)."""
        await real_fail(app_, job, msg, detail)
        if job is expired:
            await jobs.cancel_user_jobs(app_, USER_B, texts.ACCESS_REMOVED)

    monkeypatch.setattr(jobs, "fail", fail_and_cancel_b)
    ran = []
    monkeypatch.setattr(pipeline, "run", lambda url, *a, **k: ran.append(url) or (_ for _ in ()).throw(
        pipeline.PipelineError("x")))
    task = asyncio.create_task(jobs.worker(app))
    await state.queue.put(state.Job("https://youtu.be/ccccccccccc", chat_id=USER_C, status_id=9, user_id=USER_C,
                                request_id=db.add_request(USER_C, "u", "summary")))
    await asyncio.wait_for(state.queue.join(), 5)
    assert not task.done()  # still running
    task.cancel()
    assert ran == ["https://youtu.be/ccccccccccc"]  # the next job was processed
    assert state.user_jobs.get(USER_A, 0) == 0 and state.user_jobs.get(USER_B, 0) == 0


async def test_database_error_while_selecting_does_not_kill_the_worker(app, monkeypatch):
    calls = []
    real_next = jobs.next_job

    async def flaky_next(app_):
        """The first selection hits a locked database."""
        calls.append(1)
        if len(calls) == 1:
            raise sqlite3.OperationalError("database is locked")
        return await real_next(app_)

    monkeypatch.setattr(jobs, "next_job", flaky_next)
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda s: real_sleep(0))
    monkeypatch.setattr(pipeline, "run", lambda *a, **k: (_ for _ in ()).throw(pipeline.PipelineError("x")))
    access.set_state(USER_A, "allowed")
    task = asyncio.create_task(jobs.worker(app))
    await state.queue.put(state.Job("https://youtu.be/aaaaaaaaaaa", chat_id=USER_A, status_id=1, user_id=USER_A,
                                request_id=db.add_request(USER_A, "u", "summary")))
    await asyncio.wait_for(state.queue.join(), 5)
    assert not task.done()
    task.cancel()


def test_ending_a_job_twice_frees_the_slot_once():
    job = state.Job("u", chat_id=1, status_id=1, user_id=USER_A, request_id=5)
    state.jobs[5] = job
    state.user_jobs[USER_A] = 2  # this job plus another one
    jobs.end_job(job)
    jobs.end_job(job)
    assert state.user_jobs[USER_A] == 1
