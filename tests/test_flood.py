"""Strangers, pending and blocked users can't flood the bot or the admins."""
import access
import bot

from conftest import ADMIN_ID, msg_update, send


async def test_pending_user_is_reminded_at_most_every_ten_minutes(app, telegram, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(bot.time, "monotonic", lambda: clock[0])
    access.set_state(50, "pending")
    for _ in range(5):
        await send(app, msg_update(50, "hello?"))
    assert len(telegram.texts()) == 1
    clock[0] += bot.PENDING_REPLY_EVERY
    await send(app, msg_update(50, "hello?"))
    assert len(telegram.texts()) == 2


async def test_blocked_user_is_silently_ignored(app, telegram, caplog):
    access.set_state(50, "blocked")
    for text in ("/start", "hi", "https://youtu.be/abcdefghijk"):
        await send(app, msg_update(50, text))
    assert telegram.texts() == [] and "50" not in caplog.text


async def test_pending_requests_are_capped(app, telegram):
    for uid in range(100, 100 + bot.MAX_PENDING):
        access.set_state(uid, "pending")
    await send(app, msg_update(500, "/start"))
    assert access.state(500) is None
    assert "isn't accepting new access requests" in telegram.texts()[-1]
    assert not [d for d in telegram.sent("sendMessage") if d["chat_id"] == ADMIN_ID]
