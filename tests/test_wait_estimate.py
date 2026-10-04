"""The first reply shows an estimated wait, never the queue position."""
import access
from summarizer import config, stats

from conftest import msg_update, send
from tgbot import limits, state

LINK = "https://youtu.be/abcdefghijk"


async def test_nothing_ahead_means_working(app, telegram):
    access.set_state(60, "allowed")
    await send(app, msg_update(60, LINK))
    assert telegram.texts()[-1] == "⏳ Got it, working…"


async def test_jobs_ahead_give_an_estimated_wait_without_a_position(app, telegram, monkeypatch):
    for uid in (60, 61, 62):
        access.set_state(uid, "allowed")
    stats.record("job", 90)
    monkeypatch.setattr(config, "WORKERS", 1)
    monkeypatch.setattr(state, "running", {object()})  # one job running
    await send(app, msg_update(60, LINK))  # 1 ahead
    await send(app, msg_update(61, LINK))  # 2 ahead
    first, second = telegram.texts()[-2:]
    assert first == "⏳ Got it, you're in the queue. Estimated wait: about 1.5 min."
    assert second == "⏳ Got it, you're in the queue. Estimated wait: about 3 min."
    assert "position" not in first + second


def test_default_until_measured():
    assert stats.get("job", limits.JOB_SECONDS_DEFAULT) == 60



async def test_free_workers_start_right_away_and_busy_ones_share_the_wait(app, telegram, monkeypatch):
    for uid in (60, 61):
        access.set_state(uid, "allowed")
    stats.record("job", 120)
    monkeypatch.setattr(config, "WORKERS", 4)
    monkeypatch.setattr(state, "running", {object(), object(), object()})  # one worker free
    await send(app, msg_update(60, LINK))
    assert telegram.texts()[-1] == "⏳ Got it, working…"
    state.running.add(object())  # all four busy, one queued already
    await send(app, msg_update(61, LINK))
    assert telegram.texts()[-1] == "⏳ Got it, you're in the queue. Estimated wait: about 1 min."  # 2 × 120 s / 4
