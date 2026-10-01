"""A first-time requester of a video someone else already summarized can't tell it came from the cache."""
import asyncio

import pytest

import access
import bot
from summarizer import db, pipeline

from conftest import ADMIN_ID

STATS = {"steps": [["lookup", 2.0], ["Whisper", 20.0], ["summary", 8.0]], "total": 30.0, "llm": "Codex (gpt-test)"}


def test_delay_is_half_the_original_capped():
    assert bot.cached_delay(STATS) == 15
    assert bot.cached_delay({**STATS, "total": 1000}) == bot.CACHED_DELAY_MAX
    assert bot.cached_delay({}) == 0  # old summaries without timings


def _cached_result(url="https://youtu.be/abcdefghijk") -> pipeline.Result:
    """A summary that came from the cache."""
    summary = {"title": "T", "is_clickbait": False, "clickbait_answer": "", "summary": "S", "_stats": STATS}
    return pipeline.Result("youtube", url[-11:], url, {"title": "Video", "duration": 600}, "", "whisper-small", "en",
                           summary, cached=True)


@pytest.fixture
def fast_replay(monkeypatch):
    """Runs replays 100x faster and makes the pipeline answer from the cache."""
    real_sleep = asyncio.sleep
    monkeypatch.setattr(bot.asyncio, "sleep", lambda s: real_sleep(s / 100))
    monkeypatch.setattr(pipeline, "run", lambda url, *a, **k: _cached_result(url))


async def _run(app, jobs):
    """Runs the worker over the jobs and waits until the replays are delivered too."""
    for job in jobs:
        await bot.queue.put(job)
    task = asyncio.create_task(bot.worker(app))
    await asyncio.wait_for(bot.queue.join(), 5)
    for _ in range(200):
        if not bot._delayed:
            break
        await asyncio.sleep(0.02)
    task.cancel()


def _job(uid, n=1):
    """A job for `uid` with a request row."""
    job = bot.Job(f"https://youtu.be/{'a' * 10}{n}", chat_id=uid, status_id=n, user_id=uid,
                  request_id=db.add_request(uid, "u", "summary"))
    bot._jobs[job.request_id] = job
    bot._user_jobs[uid] += 1
    return job


async def test_first_time_requester_gets_a_staged_replay(app, telegram, fast_replay):
    access.set_state(60, "allowed")
    await _run(app, [_job(60)])
    stages = [d["text"].split("\n")[1] for d in telegram.sent("editMessageText") if d["chat_id"] == 60]
    assert any("Transcribing" in s for s in stages) and any("Summarizing with Codex" in s for s in stages)
    summary = telegram.sent("sendMessage")[-1]["text"]
    assert "cache" not in summary and "⏱ 15 s total" in summary
    assert db.recent_requests(60)[0]["status"] == "done" and bot._user_jobs == {}


async def test_admin_and_repeat_requesters_get_it_at_once(app, telegram, fast_replay):
    await _run(app, [_job(ADMIN_ID)])
    assert "from cache" in telegram.sent("sendMessage")[-1]["text"]
    assert telegram.sent("editMessageText") == []  # no replay


async def test_replay_does_not_block_the_queue(app, telegram, monkeypatch):
    real_sleep = asyncio.sleep
    monkeypatch.setattr(bot.asyncio, "sleep", lambda s: real_sleep(s / 10))  # 1.5 s replay
    calls = []
    monkeypatch.setattr(pipeline, "run", lambda url, *a, **k: calls.append(url[-1]) or _cached_result(url))
    for uid in (60, 70):
        access.set_state(uid, "allowed")
    jobs = [_job(60, 1), _job(70, 2)]
    for job in jobs:
        await bot.queue.put(job)
    task = asyncio.create_task(bot.worker(app))
    await asyncio.wait_for(bot.queue.join(), 1)  # both picked up long before the first replay ends
    assert calls == ["1", "2"] and len(bot._delayed) == 2
    for t in list(bot._delayed.values()):
        t.cancel()
    task.cancel()


async def test_removed_user_replay_is_cancelled(app, telegram, monkeypatch):
    real_sleep = asyncio.sleep
    monkeypatch.setattr(bot.asyncio, "sleep", lambda s: real_sleep(s / 10))
    monkeypatch.setattr(pipeline, "run", lambda url, *a, **k: _cached_result(url))
    access.set_state(60, "allowed")
    await bot.queue.put(_job(60))
    task = asyncio.create_task(bot.worker(app))
    await asyncio.wait_for(bot.queue.join(), 1)
    await bot.cancel_user_jobs(app, 60, bot.ACCESS_REMOVED)
    await real_sleep(0.1)
    task.cancel()
    assert bot._delayed == {} and bot._user_jobs == {}
    assert not [d for d in telegram.sent("sendMessage") if d["chat_id"] == 60]  # summary never sent
