"""The Telegram side, through a real PTB Application and a fake Telegram."""
import asyncio

import pytest

import access
import bot
from summarizer import db, pipeline, updates

from conftest import ADMIN_ID, callback_update, msg_update, send

STRANGER, FRIEND = 50, 60


# ---------- access flows ----------

async def test_stranger_message_is_told_to_send_start(app, telegram):
    await send(app, msg_update(STRANGER, "hello"))
    assert telegram.texts() == ["🔒 This is a private bot. Send /start to request access."]
    assert access.state(STRANGER) is None


async def test_start_files_one_request_and_notifies_admins_once(app, telegram):
    await send(app, msg_update(STRANGER, "/start", name="Eve", username="eve"))
    await send(app, msg_update(STRANGER, "/start"))
    assert access.state(STRANGER) == "pending"
    to_admin = [d for d in telegram.sent("sendMessage") if d["chat_id"] == ADMIN_ID]
    assert len(to_admin) == 1 and "Eve (@eve, 50)" in to_admin[0]["text"]
    assert "waiting for the admin" in telegram.texts()[-1]


async def test_admin_allows_and_user_is_welcomed(app, telegram):
    access.set_state(STRANGER, "pending", "Eve", None)
    await send(app, callback_update(ADMIN_ID, f"allow:{STRANGER}"))
    assert access.state(STRANGER) == "allowed"
    assert any(d["chat_id"] == STRANGER and "You now have access" in d["text"] for d in telegram.sent("sendMessage"))


async def test_non_admin_cannot_press_admin_buttons(app, telegram):
    access.set_state(FRIEND, "allowed")
    await send(app, callback_update(FRIEND, f"allow:{STRANGER}"))
    assert access.state(STRANGER) is None
    assert telegram.sent("answerCallbackQuery")[-1].get("text") == "Admins only."


async def test_blocked_user_gets_no_reply(app, telegram):
    access.set_state(STRANGER, "blocked")
    await send(app, msg_update(STRANGER, "/start"))
    assert telegram.texts() == []


async def test_users_lists_admins_and_groups(app, telegram):
    access.set_state(FRIEND, "allowed", "Ana", "ana")
    access.set_state(STRANGER, "pending", "Eve", None)
    await send(app, msg_update(ADMIN_ID, "/users"))
    texts = "\n".join(telegram.texts())
    assert "Admins" in texts and "Ana (@ana, 60)" in texts and "Eve (50)" in texts


# ---------- links, history, models ----------

async def test_link_from_allowed_user_is_queued_and_logged(app, telegram):
    access.set_state(FRIEND, "allowed")
    await send(app, msg_update(FRIEND, "check https://youtu.be/abcdefghijk"))
    assert bot.queue.qsize() == 1
    assert telegram.texts()[-1].startswith("⏳ Got it")
    assert db.recent_requests(FRIEND)[0]["url"] == "https://youtu.be/abcdefghijk"


async def test_history_shows_users_only_their_own(app, telegram):
    access.set_state(FRIEND, "allowed")
    db.add_request(ADMIN_ID, "https://admin-link", "summary")
    db.add_request(FRIEND, "https://friend-link", "summary")
    await send(app, msg_update(FRIEND, "/history"))
    assert "friend-link" in telegram.texts()[-1] and "admin-link" not in telegram.texts()[-1]


async def test_models_picker_two_steps(app, telegram, codex_home):
    await send(app, msg_update(ADMIN_ID, "/models"))
    first = telegram.sent("sendMessage")[-1]
    assert "Choose a provider" in first["text"]
    await send(app, callback_update(ADMIN_ID, "llm:b:codex"))
    assert "Codex models" in telegram.sent("editMessageText")[-1]["text"]
    await send(app, callback_update(ADMIN_ID, "llm:m:codex:gpt-a"))
    assert db.get_user_llm(ADMIN_ID) == ("codex", "gpt-a")


# ---------- output ----------

def _result(summary_text: str, cached=False, stats=None) -> pipeline.Result:
    """A finished pipeline result with the given summary text."""
    summary = {"title": "<b>T</b>", "is_clickbait": False, "clickbait_answer": "", "summary": summary_text}
    if stats:
        summary["_stats"] = stats
    return pipeline.Result("youtube", "id", "https://youtu.be/id", {"title": "T"}, "", "captions", "en",
                           summary, cached=cached)


def test_render_escapes_html_and_splits_long_messages():
    chunks = bot.render(_result("• point & <i>\n" * 600))
    assert len(chunks) > 1 and all(len(c) <= bot.TG_LIMIT for c in chunks)
    assert "&lt;b&gt;T&lt;/b&gt;" in chunks[0] and "&amp; &lt;i&gt;" in chunks[0]


def test_footer_hides_cache_from_first_time_requesters():
    stats = {"steps": [["lookup", 2.0], ["summary", 9.0]], "total": 11.0, "llm": "Codex (gpt-test)"}
    assert "from cache" in bot.details(_result("x", cached=True, stats=stats), reveal_cache=True)
    hidden = bot.details(_result("x", cached=True, stats=stats), reveal_cache=False)
    assert "cache" not in hidden and "⏱" not in hidden
    assert "11 s total" in bot.details(_result("x", stats=stats))


@pytest.mark.parametrize("sec,text", [(3, "5 s"), (42, "45 s"), (65, "1 min"), (95, "1.5 min"), (900, "15 min")])
def test_eta_format(sec, text):
    assert bot._fmt_eta(sec) == text


async def test_progress_edits_in_order_and_stops(app, telegram, monkeypatch):
    monkeypatch.setattr(bot.Progress, "TICK", 0.05)
    job = bot.Job("u", chat_id=FRIEND, status_id=7)
    progress = bot.Progress(app, asyncio.get_running_loop(), job)
    await asyncio.to_thread(progress, "step one", 30)
    await asyncio.sleep(0.1)  # stages arrive seconds apart in practice
    await asyncio.to_thread(progress, "step two", 20)
    await asyncio.sleep(0.2)
    await progress.close()
    edits = [d["text"] for d in telegram.sent("editMessageText")]
    assert edits and edits[-1].startswith("step two")
    assert [e.split("\n")[0] for e in edits].index("step two") > [e.split("\n")[0] for e in edits].index("step one")
    count = len(edits)
    await asyncio.sleep(0.15)
    assert len(telegram.sent("editMessageText")) == count  # nothing after close()


async def test_update_notification_is_sent_once_per_version(app, telegram, monkeypatch):
    monkeypatch.setattr(updates, "check", lambda: [{"tool": "Codex", "installed": "1.0.0", "latest": "1.1.0",
                                                     "command": "sudo npm install -g @openai/codex"}])
    rounds = iter([None, None, None])

    async def fake_sleep(_):
        """Lets the checker loop three times, then stops it."""
        if next(rounds, "stop") == "stop":
            raise asyncio.CancelledError

    monkeypatch.setattr(bot.asyncio, "sleep", fake_sleep)
    with pytest.raises(asyncio.CancelledError):
        await bot.update_checker(app)
    notes = [d for d in telegram.sent("sendMessage") if d["chat_id"] == ADMIN_ID]
    assert len(notes) == 1 and "<pre>sudo npm install -g @openai/codex</pre>" in notes[0]["text"]
