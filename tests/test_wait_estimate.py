"""The first reply shows an estimated wait, never the queue position."""
import access
import bot
from summarizer import stats

from conftest import msg_update, send

LINK = "https://youtu.be/abcdefghijk"


async def test_nothing_ahead_means_working(app, telegram):
    access.set_state(60, "allowed")
    await send(app, msg_update(60, LINK))
    assert telegram.texts()[-1] == "⏳ Got it, working…"


async def test_jobs_ahead_give_an_estimated_wait_without_a_position(app, telegram, monkeypatch):
    for uid in (60, 61, 62):
        access.set_state(uid, "allowed")
    stats.record("job", 90)
    monkeypatch.setattr(bot, "_running", object())  # one job running
    await send(app, msg_update(60, LINK))  # 1 ahead
    await send(app, msg_update(61, LINK))  # 2 ahead
    first, second = telegram.texts()[-2:]
    assert first == "⏳ Got it, you're in the queue. Estimated wait: about 1.5 min."
    assert second == "⏳ Got it, you're in the queue. Estimated wait: about 3 min."
    assert "position" not in first + second


def test_default_until_measured():
    assert stats.get("job", bot.JOB_SECONDS_DEFAULT) == 60
