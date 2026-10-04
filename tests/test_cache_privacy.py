"""A first-time requester of reused work can't tell it came from the cache: every cached answer is replayed."""
import asyncio

import pytest

import access
from summarizer import db, pipeline

from conftest import ADMIN_ID
from summarizer import stats
import time
from tgbot import jobs, lifecycle, sending, state, texts

STATS = {"steps": [["lookup", 2.0], ["Whisper", 20.0], ["summary", 8.0]], "total": 30.0, "llm": "Codex (gpt-test)"}
URL = "https://youtu.be/abcdefghijk"


def _cached_result(url=URL, *, stats=STATS, source="whisper-small", summary=True) -> pipeline.Result:
    """An answer that came from the cache."""
    s = {"title": "T", "is_clickbait": False, "clickbait_answer": "", "summary": "S", "_stats": stats}
    return pipeline.Result("youtube", url[-11:], url, {"title": "Video", "duration": 600}, "word " * 100,
                           source, "en", s if summary else None, cached=True)


def _planned(transcript_only=False, **kw) -> pipeline.Result:
    """A cached result with its replay planned, as pipeline.run does for a first-time requester."""
    r = _cached_result(**kw)
    pipeline.plan_replay(r, transcript_only, "codex")
    return r


# ---------- the replay plan ----------

def test_delay_is_half_the_original_capped():
    r = _planned()
    assert r.replay_total == 15 and r.replay_steps == [("lookup", 1.0), ("Whisper", 10.0), ("summary", 4.0)]
    long = _planned(stats={**STATS, "steps": [["lookup", 2.0], ["Whisper", 2000.0], ["summary", 8.0]]})
    assert long.replay_total == pipeline.REPLAY_MAX
    assert sum(sec for _, sec in long.replay_steps) == pytest.approx(pipeline.REPLAY_MAX)  # footer agrees


def test_old_summary_without_timings_is_estimated():
    r = _planned(stats=None, source="captions")
    assert [n for n, _ in r.replay_steps] == ["lookup", "captions", "summary"] and r.replay_total > 0


def test_summary_written_from_a_reused_transcript_gets_its_transcript_step():
    r = _planned(stats={"steps": [["lookup", 2.0], ["summary", 8.0]], "total": 10.0}, source="whisper-small")
    assert [n for n, _ in r.replay_steps] == ["lookup", "Whisper", "summary"]


def test_transcript_replay_has_no_summary_step():
    r = _planned(transcript_only=True)
    assert [n for n, _ in r.replay_steps] == ["lookup", "Whisper"]
    bare = _planned(transcript_only=True, summary=False, source="none")  # no summary, and no speech found
    assert [n for n, _ in bare.replay_steps] == ["lookup", "Whisper"]


def test_pipeline_plans_a_replay_only_for_first_time_requesters(fake_media, llm):
    pipeline.run(URL, lambda *a: None)  # the admin summarized it
    first = db.add_request(60, URL, "transcript")
    r = pipeline.run(URL, lambda *a: None, transcript_only=True, request_id=first, hide_cache_from=60)
    assert r.cached and r.replay_steps and "summary" not in [n for n, _ in r.replay_steps]
    db.update_request(first, status="done")
    again = db.add_request(60, URL, "summary")
    r = pipeline.run(URL, lambda *a: None, request_id=again, hide_cache_from=60)
    assert r.cached and r.replay_steps is None  # they've had this video before
    assert pipeline.run(URL, lambda *a: None).replay_steps is None  # admins


# ---------- the bot plays it back ----------

@pytest.fixture
def fast_replay(monkeypatch):
    """Runs replays 100x faster; the pipeline answers from the cache, planning a replay like the real one."""
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda s: real_sleep(s / 100))

    def run(url, *a, hide_cache_from=None, transcript_only=False, **k):
        """A cached answer; replayed for non-admins (none of these users saw the video before)."""
        r = _cached_result(url)
        if hide_cache_from:
            pipeline.plan_replay(r, transcript_only, "codex")
        return r

    monkeypatch.setattr(pipeline, "run", run)


async def _run(app, queued):
    """Runs the worker over the queued jobs and waits until the replays are delivered too."""
    for job in queued:
        await state.queue.put(job)
    task = asyncio.create_task(jobs.worker(app))
    await asyncio.wait_for(state.queue.join(), 5)
    for _ in range(200):
        if not state.delayed:
            break
        await asyncio.sleep(0.02)
    task.cancel()


def _job(uid, n=1, **kw):
    """A job for `uid` with a request row."""
    job = state.Job(f"https://youtu.be/{'a' * 10}{n}", chat_id=uid, status_id=n, user_id=uid,
                  request_id=db.add_request(uid, "u", "summary"), **kw)
    state.jobs[job.request_id] = job
    state.user_jobs[uid] += 1
    return job


async def test_first_time_requester_gets_a_staged_replay(app, telegram, fast_replay):
    access.set_state(60, "allowed")
    await _run(app, [_job(60)])
    stages = [d["text"].split("\n")[1] for d in telegram.sent("editMessageText") if d["chat_id"] == 60]
    assert any("Transcribing" in s for s in stages) and any("Summarizing with Codex" in s for s in stages)
    summary = telegram.sent("sendMessage")[-1]["text"]
    assert "cache" not in summary and "⏱ 15 s total: lookup 1 s · Whisper 10 s · summary 4 s" in summary
    assert db.recent_requests(60)[0]["status"] == "done" and state.user_jobs == {}


async def test_cached_transcript_is_replayed_too(app, telegram, fast_replay):
    access.set_state(60, "allowed")
    await _run(app, [_job(60, transcript_only=True)])
    stages = [d["text"] for d in telegram.sent("editMessageText") if d["chat_id"] == 60]
    assert any("Transcribing" in s for s in stages) and not any("Summarizing" in s for s in stages)
    assert telegram.sent("sendDocument") and db.recent_requests(60)[0]["status"] == "done"


async def test_admin_and_repeat_requesters_get_it_at_once(app, telegram, fast_replay):
    await _run(app, [_job(ADMIN_ID)])
    assert "from cache" in telegram.sent("sendMessage")[-1]["text"]
    assert telegram.sent("editMessageText") == []  # no replay


async def test_replay_does_not_block_the_queue(app, telegram, monkeypatch):
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda s: real_sleep(s / 10))  # 1.5 s replay
    calls = []
    monkeypatch.setattr(pipeline, "run", lambda url, *a, **k: calls.append(url[-1]) or _planned())
    for uid in (60, 70):
        access.set_state(uid, "allowed")
    for job in [_job(60, 1), _job(70, 2)]:
        await state.queue.put(job)
    task = asyncio.create_task(jobs.worker(app))
    await asyncio.wait_for(state.queue.join(), 1)  # both picked up long before the first replay ends
    assert calls == ["1", "2"] and len(state.delayed) == 2
    for t in list(state.delayed.values()):
        t.cancel()
    task.cancel()


async def test_removed_user_replay_is_cancelled(app, telegram, monkeypatch):
    real_sleep = asyncio.sleep
    monkeypatch.setattr(asyncio, "sleep", lambda s: real_sleep(s / 10))
    monkeypatch.setattr(pipeline, "run", lambda url, *a, **k: _planned())
    access.set_state(60, "allowed")
    await state.queue.put(_job(60))
    task = asyncio.create_task(jobs.worker(app))
    await asyncio.wait_for(state.queue.join(), 1)
    await jobs.cancel_user_jobs(app, 60, texts.ACCESS_REMOVED)
    await real_sleep(0.1)
    task.cancel()
    assert state.delayed == {} and state.user_jobs == {}
    assert not [d for d in telegram.sent("sendMessage") if d["chat_id"] == 60]  # summary never sent


# ---------- pacing off the worker (hold) ----------

def _held(url, seconds=0.6) -> pipeline.Result:
    """A fresh summary written from reused work for a first-time requester: `seconds` still owed."""
    r = _cached_result(url, stats={**STATS, "total": 9.0})
    r.cached = False
    r.hold = [("🎬 Video (10:00)\n✅ Transcript ready\n🧠 Summarizing with Codex (gpt-test)…", seconds)]
    r.replay_steps, r.replay_total = [("lookup", 1.0), ("Whisper", 3.0), ("summary", 5.0)], 9.0 + seconds
    return r


async def test_a_hold_does_not_block_the_next_user(app, telegram, monkeypatch):
    for uid in (60, 70):
        access.set_state(uid, "allowed")
    monkeypatch.setattr(pipeline, "run", lambda url, *a, **k: _held(url) if url.endswith("1") else _cached_result(url))
    await _run(app, [_job(60, 1), _job(ADMIN_ID, 2)])
    to = [m["chat_id"] for m in telegram.sent("sendMessage")]
    assert to.index(ADMIN_ID) < to.index(60)  # the second job was answered while the first one held
    held = [d["text"] for d in telegram.sent("editMessageText") if d["chat_id"] == 60]
    assert any(t.startswith("🎬 Video (10:00)\n✅ Transcript ready\n🧠 Summarizing") for t in held)
    footer = [m["text"] for m in telegram.sent("sendMessage") if m["chat_id"] == 60][-1]
    assert "⏱ 10 s total: lookup 1 s · Whisper 3 s · summary 5 s" in footer and "cache" not in footer
    req = db.recent_requests(60)[0]
    assert (req["status"], req["cached"]) == ("done", 0)
    assert stats.get("job", 0) == 9.0  # the worker's time, without the pause
    assert state.user_jobs == {}


async def test_elapsed_time_continues_through_the_hold(app):
    job = _job(60)
    job.started_at = time.monotonic() - 100
    progress = sending.Progress(app, asyncio.get_running_loop(), job)
    try:
        assert time.monotonic() - progress.started >= 100
    finally:
        await progress.close()


async def test_cancel_during_a_hold_frees_the_slot_once(app, telegram, monkeypatch):
    access.set_state(60, "allowed")
    monkeypatch.setattr(pipeline, "run", lambda url, *a, **k: _held(url, seconds=30))
    job = _job(60)
    other = _job(60, 2)  # a second job of the same user, still queued
    await state.queue.put(job)
    task = asyncio.create_task(jobs.worker(app))
    for _ in range(200):
        if state.delayed:
            break
        await asyncio.sleep(0.01)
    await jobs.cancel_job(app, job, texts.ACCESS_REMOVED)
    await asyncio.sleep(0.05)
    task.cancel()
    assert state.user_jobs[60] == 1 and other.request_id in state.jobs  # only the held job's slot was freed
    assert not [m for m in telegram.sent("sendMessage") if m["chat_id"] == 60]


async def test_shutdown_during_a_hold_tells_the_user(app, telegram, monkeypatch):
    access.set_state(60, "allowed")
    monkeypatch.setattr(pipeline, "run", lambda url, *a, **k: _held(url, seconds=30))
    await state.queue.put(_job(60))
    app.bot_data["workers"] = [asyncio.create_task(jobs.worker(app))]
    for _ in range(200):
        if state.delayed:
            break
        await asyncio.sleep(0.01)
    await lifecycle.post_stop(app)
    assert telegram.sent("editMessageText")[-1]["text"].startswith(texts.STOPPED + "\n") and state.jobs == {}
