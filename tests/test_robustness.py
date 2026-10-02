"""Robustness: edited messages, malformed buttons, /models off the event loop, status stages in order."""
import asyncio
import threading

from summarizer import summarize

from conftest import ADMIN_ID, callback_update, msg_update, send
from tgbot import handlers, sending, state


async def test_edited_commands_and_messages_are_ignored(app, telegram):
    await send(app, msg_update(ADMIN_ID, "/help", edited=True))
    await send(app, msg_update(ADMIN_ID, "https://youtu.be/abcdefghijk", edited=True))
    assert telegram.texts() == [] and state.queue.qsize() == 0


async def test_malformed_model_buttons_show_the_first_step(app, telegram, codex_home):
    for data in ("llm:b", "llm:m:codex", "llm:b:nonexistent", "llm:m:nope:model", "llm:"):
        await send(app, callback_update(ADMIN_ID, data))
    edits = telegram.sent("editMessageText")
    assert len(edits) == 5 and all("Choose a provider" in e["text"] for e in edits)


async def test_model_lists_are_fetched_off_the_event_loop(monkeypatch):
    seen = []

    def list_models(backend):
        """Records which thread asked."""
        seen.append(threading.current_thread() is threading.main_thread())
        return [{"id": "m", "name": "M", "description": ""}]

    monkeypatch.setattr(summarize, "list_models", list_models)
    await handlers.llm_models(ADMIN_ID, "codex")
    assert seen == [False]


def test_api_model_lists_are_cached(monkeypatch):
    calls = []
    monkeypatch.setattr(summarize, "_api_models_cache", {})
    monkeypatch.setattr(summarize, "_openai_models", lambda: calls.append(1) or [{"id": "gpt-x"}])
    assert summarize.list_models("openai-api") == summarize.list_models("openai-api") == [{"id": "gpt-x"}]
    assert len(calls) == 1


async def test_every_stage_is_shown_even_when_stages_come_fast(app, telegram):
    job = state.Job("u", chat_id=60, status_id=7)
    progress = sending.Progress(app, asyncio.get_running_loop(), job)
    await asyncio.to_thread(lambda: (progress("step one", 30), progress("step two", 20)))
    await asyncio.sleep(0.1)
    await progress.close()
    firsts = [d["text"].split("\n")[0] for d in telegram.sent("editMessageText")]
    assert firsts == ["step one", "step two"]
