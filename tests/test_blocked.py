"""YouTube/TikTok blocking the server is recognized, reported to the user, and the admins are told."""
import asyncio

import pytest

import access
from summarizer import media, pipeline

from conftest import ADMIN_ID
from tgbot import jobs, state

LINK = "https://youtu.be/abcdefghijk"


@pytest.mark.parametrize("text,blocked", [
    ("[youtube] x: Sign in to confirm you're not a bot. Use --cookies", True),
    ("[youtube] x: Sign in to confirm you’re not a bot", True),
    ("HTTP Error 429: Too Many Requests", True),
    ("[TikTok] 123: Your IP address is blocked from accessing this post", True),
    ("[youtube] x: Sign in to confirm your age", False),
    ("[youtube] x: Video unavailable", False),
])
def test_block_patterns(text, blocked):
    assert media._is_blocked(text) is blocked


def _blocked(*a, **k):
    """A download refused by the platform."""
    raise media.Blocked("HTTP Error 429: Too Many Requests")


def test_block_during_lookup_is_reported(monkeypatch, llm):
    monkeypatch.setattr(media, "probe", _blocked)
    with pytest.raises(pipeline.Blocked, match="YouTube is currently blocking downloads") as e:
        pipeline.run(LINK, lambda *a: None)
    assert e.value.platform == "youtube" and "429" in e.value.detail


def test_block_during_captions_is_not_swallowed(monkeypatch, llm):
    from helpers import meta
    monkeypatch.setattr(media, "probe", lambda v, *a, **k: meta())
    monkeypatch.setattr(media, "_ytdlp", _blocked)  # fetch_captions catches other errors and carries on
    with pytest.raises(pipeline.Blocked):
        pipeline.run(LINK, lambda *a: None)
    assert llm == []  # never got to summarizing without a transcript


async def test_admins_hear_about_a_block_once(app, telegram, monkeypatch):
    access.set_state(60, "allowed")
    monkeypatch.setattr(pipeline, "run", lambda *a, **k: (_ for _ in ()).throw(
        pipeline.Blocked("🚫 YouTube is currently blocking downloads…", detail="HTTP Error 429", platform="youtube")))
    for n in range(2):
        await state.queue.put(state.Job(LINK, chat_id=60, status_id=n, user_id=60, request_id=n + 1))
    task = asyncio.create_task(jobs.worker(app))
    await asyncio.wait_for(state.queue.join(), 5)
    task.cancel()
    to_user = [d["text"] for d in telegram.sent("editMessageText") if d["chat_id"] == 60]
    assert len(to_user) == 2 and all(t.startswith("🚫 YouTube is currently blocking") for t in to_user)
    to_admin = [d["text"] for d in telegram.sent("sendMessage") if d["chat_id"] == ADMIN_ID]
    assert len(to_admin) == 1 and "HTTP Error 429" in to_admin[0]


async def test_the_admin_notice_names_the_platform_properly(app, telegram):
    from summarizer import pipeline
    from tgbot import jobs as jobs_
    await jobs_._notify_admins_of_block(app, pipeline.Blocked("🚫", detail="Sign in", platform="youtube"))
    assert telegram.sent("sendMessage")[-1]["text"].startswith("🚫 YouTube is blocking downloads")
