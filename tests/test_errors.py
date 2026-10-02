"""Users only ever see expected messages; admins get the technical details."""
import asyncio

import pytest

import access
from summarizer import media, pipeline

from conftest import ADMIN_ID
from tgbot import jobs, state, texts

USER = 60


@pytest.fixture(autouse=True)
def approved_user():
    """The test user is an approved user (the worker cancels jobs of users without access)."""
    access.set_state(USER, "allowed", "Ana", None)


@pytest.mark.parametrize("raw,expected", [
    ("[youtube] x: Private video. Sign in if you've been granted access", "private"),
    ("[youtube] x: Sign in to confirm your age. This video may be inappropriate", "age-restricted"),
    ("[youtube] x: The uploader has not made this video available in your country", "country"),
    ("[youtube] x: Video unavailable. This video has been removed by the uploader", "unavailable"),
    ("the download timed out", "took too long"),
    ("[generic] something nobody expected", "Couldn't load this video"),
])
def test_download_errors_become_fixed_messages(raw, expected):
    text = media.describe(media.MediaError(raw))
    assert expected in text and "[youtube]" not in text and "[generic]" not in text


async def _run_one(app, outcome, user=USER):
    """Runs a single job through the worker with the pipeline replaced by `outcome` (a raising function)."""
    monkey = pytest.MonkeyPatch()
    monkey.setattr(pipeline, "run", lambda *a, **k: outcome())
    try:
        await state.queue.put(state.Job("https://youtu.be/abcdefghijk", chat_id=user, status_id=5, user_id=user,
                                    request_id=1))
        task = asyncio.create_task(jobs.worker(app))
        await asyncio.wait_for(state.queue.join(), 5)
        task.cancel()
    finally:
        monkey.undo()


def _boom():
    """An unexpected bug inside the pipeline."""
    raise KeyError("secret internal detail /home/x")


async def test_unexpected_error_for_a_user_is_generic_and_admins_are_told_once(app, telegram):
    access.set_state(USER, "allowed", "Ana", None)
    await _run_one(app, _boom)
    await _run_one(app, _boom)
    to_user = [d["text"] for m, d in telegram.calls if m in ("editMessageText", "sendMessage") and d["chat_id"] == USER]
    assert to_user == [texts.INTERNAL_ERROR_NOTIFIED] * 2
    to_admin = [d["text"] for d in telegram.sent("sendMessage") if d["chat_id"] == ADMIN_ID]
    assert len(to_admin) == 1 and "KeyError" in to_admin[0] and "Ana (60)" in to_admin[0]


async def test_admin_sees_details_inline(app, telegram):
    await _run_one(app, _boom, user=ADMIN_ID)
    text = telegram.sent("editMessageText")[-1]["text"]
    assert text.startswith(texts.INTERNAL_ERROR) and "Details: KeyError" in text


async def test_expected_error_hides_detail_from_users(app, telegram):
    def fail():
        """A download failure with raw yt-dlp output as detail."""
        raise pipeline.PipelineError("This video is unavailable.", detail="[youtube] x: ERROR raw output")
    await _run_one(app, fail)
    text = telegram.sent("editMessageText")[-1]["text"]
    assert text == "⚠️ This video is unavailable." and "raw output" not in text
