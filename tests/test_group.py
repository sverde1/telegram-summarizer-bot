"""Several links in one message: parsing, refusals, one job with per-link results, cancel/retry, history."""
import asyncio
import re

import pytest

import access
from summarizer import config, db, pipeline
from summarizer.urls import find_urls

from conftest import callback_update, msg_update, send
from tgbot import jobs, state

ANA = 60
A, B, C = "https://youtu.be/aaaaaaaaaaa", "https://youtu.be/bbbbbbbbbbb", "https://youtu.be/ccccccccccc"
SUMMARY = {"title": "T", "is_clickbait": False, "clickbait_answer": "", "summary": "S",
           "_stats": {"steps": [["summary", 1.0]], "total": 1.0, "llm": "Codex (gpt-test)"}}


def test_find_urls_reads_any_layout():
    text = f"""Parts:
1. {A}
2) {B},
- {C}.
see ({A}) and https://x.y/1,https://x.y/2"""
    assert find_urls(text) == [(A, 1), (B, 2), (C, None), ("https://x.y/1", None), ("https://x.y/2", None)]
    assert find_urls(f"{A}, {B}") == [(A, None), (B, None)] and find_urls("• " + A) == [(A, None)]


@pytest.fixture
def pipe(monkeypatch):
    """An approved user and a pipeline answering per link (the link's last letter becomes the title)."""
    access.set_state(ANA, "allowed")
    calls = []

    def run(url, progress, *, request_id=None, transcript_only=False, **kw):
        """A fake run: link-specific results; "fail" in the link fails it."""
        calls.append((url, transcript_only))
        if "ccc" in url and getattr(run, "fail_c", False):
            raise pipeline.PipelineError("⚠️ This video is unavailable.")
        vid = re.search(r"youtu\.be/([\w-]{11})", url).group(1)
        db.update_request(request_id, platform="youtube", video_id=vid, status="processing")
        summary = None if transcript_only else {**SUMMARY, "title": f"Title {vid[-1]}"}
        return pipeline.Result("youtube", vid, url, {"title": f"Title {vid[-1]}"}, "[0:00] hi", "captions",
                               "en", summary)

    monkeypatch.setattr(pipeline, "run", run)
    run.calls = calls
    return run


async def _work(app):
    """Runs the worker until the queue is done."""
    task = asyncio.create_task(jobs.worker(app))
    await asyncio.wait_for(state.queue.join(), 5)
    for _ in range(100):
        if not state.delayed:
            break
        await asyncio.sleep(0.01)
    task.cancel()


async def test_unrelated_links_get_a_summary_each_with_their_own_request(app, telegram, pipe):
    await send(app, msg_update(ANA, f"{A}\n{B}"))
    assert state.queue.qsize() == 1  # one job for the message
    await _work(app)
    sent = [d for d in telegram.sent("sendMessage") if "Title" in d.get("text", "")]
    assert [re.search(r"Title (\w)", d["text"]).group(1) for d in sent] == ["a", "b"]
    rows = {r["url"]: r for r in db.recent_requests(ANA)}
    group = rows[f"{A} {B}"]
    assert group["kind"] == "group" and group["status"] == "done"
    for url in (A, B):
        assert rows[url]["status"] == "done" and rows[url]["group_id"] == group["id"]
        assert any(f"ask:{rows[url]['id']}" in str(d.get("reply_markup")) for d in sent)
    assert db.usage(ANA, "daily")[0] == 2  # the links count, the group doesn't


async def test_the_same_video_twice_is_summarized_once_and_a_failed_link_is_noted(app, telegram, pipe):
    pipe.fail_c = True
    await send(app, msg_update(ANA, f"{A} {A}/ {C}"))
    await _work(app)
    titles = [d["text"] for d in telegram.sent("sendMessage") if "Title" in d.get("text", "")]
    assert len(titles) == 1
    assert any(d["text"].startswith(f"⚠️ {C}\n") and "unavailable" in d["text"] for d in telegram.sent("sendMessage"))
    assert {r["url"]: r["status"] for r in db.recent_requests(ANA)}[C] == "failed"


async def test_whole_message_refused_when_over_the_limit_or_invalid(app, telegram, pipe, monkeypatch):
    many = " ".join(f"https://youtu.be/{'a' * 10}{i}" for i in range(11))
    await send(app, msg_update(ANA, many))
    assert "at most 10 links" in telegram.texts()[-1] and "Nothing was processed" in telegram.texts()[-1]
    await send(app, msg_update(ANA, f"{A} https://example.com/x"))
    assert "Nothing was processed" in telegram.texts()[-1]
    await send(app, msg_update(ANA, f"{A} https://drive.google.com/file/d/{'x' * 25}/view"))
    assert "Google Drive and Dropbox" in telegram.texts()[-1]
    monkeypatch.setattr(config, "DAILY_LIMIT", 3)
    db.add_request(ANA, A, "summary")
    db.add_request(ANA, A, "summary")
    await send(app, msg_update(ANA, f"{A} {B}"))
    assert "1 requests left today, and these are 2 links" in telegram.texts()[-1]
    assert state.queue.qsize() == 0 and not db.recent_requests(ANA)[0]["kind"] == "group"


async def test_cancel_closes_the_links_and_try_again_reuses_them(app, telegram, pipe):
    await send(app, msg_update(ANA, f"{A} {B}"))
    gid = next(r["id"] for r in db.recent_requests(ANA) if r["kind"] == "group")
    await send(app, callback_update(ANA, f"cancel:{gid}"))  # still queued
    assert {r["status"] for r in db.recent_requests(ANA)} == {"cancelled"}
    used = db.usage(ANA, "daily")[0]
    await send(app, callback_update(ANA, f"retry:{gid}"))
    await _work(app)
    assert {r["status"] for r in db.recent_requests(ANA)} == {"done"} and db.usage(ANA, "daily")[0] == used


async def test_history_shows_the_message_once_and_again_redoes_it(app, telegram, pipe):
    await send(app, msg_update(ANA, f"{A} {B}"))
    await _work(app)
    await send(app, msg_update(ANA, "/history"))
    history = telegram.texts()[-1]
    assert "🔗 2 links" in history and "Title a" not in history
    await send(app, msg_update(ANA, "/again"))
    job = await state.queue.get()
    assert job.job_kind is state.JobKind.GROUP and job.urls == [A, B] and not job.use_cache


async def test_a_block_stops_the_whole_message(app, telegram, pipe, monkeypatch):
    def blocked(url, progress, **kw):
        """YouTube refuses."""
        raise pipeline.Blocked("🚫 YouTube is blocking", detail="bot check", platform="youtube")

    monkeypatch.setattr(pipeline, "run", blocked)
    await send(app, msg_update(ANA, f"{A} {B}"))
    await _work(app)
    assert any("blocking" in t for t in telegram.texts())
    assert {r["status"] for r in db.recent_requests(ANA)} == {"failed"}


async def test_owed_time_for_cached_work_is_waited_out_once(app, telegram, pipe, monkeypatch):
    def cached(url, progress, *, request_id=None, transcript_only=False, **kw):
        """Cached work a first-time requester must not see as instant."""
        r = pipe(url, progress, request_id=request_id, transcript_only=transcript_only)
        r.hold = [("🧠 Summarizing…", 0.02)]
        return r

    monkeypatch.setattr(pipeline, "run", cached)
    await send(app, msg_update(ANA, f"{A} {B}"))
    await _work(app)
    assert len([d for d in telegram.sent("sendMessage") if "Title" in d.get("text", "")]) == 2


def test_a_group_job_needs_its_links():
    with pytest.raises(ValueError):
        state.Job("x", 1, 1, job_kind=state.JobKind.GROUP, urls=[A])
    state.Job("x", 1, 1, job_kind=state.JobKind.GROUP, urls=[A, B], part_ids=[1, 2])
