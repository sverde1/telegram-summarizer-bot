"""Daily limit: links per rolling 24 h, global plus per-user overrides, set and shown with /limit."""
import pytest

import access
from summarizer import config, db

from conftest import ADMIN_ID, msg_update, send
from tgbot import limits, render, state

ANA, BOB = 60, 61
LINK = "https://youtu.be/abcdefghijk"


@pytest.fixture(autouse=True)
def users(monkeypatch):
    """Two approved users; jobs are only queued, never run."""
    monkeypatch.setattr(config, "DAILY_LIMIT", 3)
    for uid in (ANA, BOB):
        access.set_state(uid, "allowed")
    db.touch_user(ANA, "Ana", "ana")  # updates existing rows only


def _use(uid, n):
    """Records n links sent by uid (any kind and outcome counts)."""
    for kind in (["summary", "again", "transcript"] * n)[:n]:
        rid = db.add_request(uid, LINK, kind)
        db.update_request(rid, status="failed")


async def test_global_limit_refuses_without_creating_a_request(app, telegram):
    _use(ANA, 3)
    await send(app, msg_update(ANA, LINK))
    assert telegram.texts()[-1].startswith("⏳ You've reached today's limit of 3 requests. You can send more in about")
    assert db.usage(ANA, "daily")[0] == 3 and state.queue.qsize() == 0


async def test_under_the_limit_is_accepted_and_counts(app, telegram):
    _use(ANA, 2)
    await send(app, msg_update(ANA, LINK))
    assert state.queue.qsize() == 1 and db.usage(ANA, "daily")[0] == 3


async def test_override_takes_precedence_both_ways(app, telegram):
    _use(ANA, 3)
    _use(BOB, 1)
    db.set_user_limit(ANA, "daily", 10)
    db.set_user_limit(BOB, "daily", 1)
    await send(app, msg_update(ANA, LINK))
    await send(app, msg_update(BOB, LINK))
    assert state.queue.qsize() == 1 and "limit of 1 requests" in telegram.texts()[-1]


async def test_zero_means_no_limit_and_admins_are_exempt(app, telegram):
    _use(ADMIN_ID, 5)
    await send(app, msg_update(ADMIN_ID, LINK))
    db.set_setting("daily_limit", "0")
    _use(ANA, 5)
    await send(app, msg_update(ANA, LINK))
    assert state.queue.qsize() == 2


async def test_admin_sets_global_and_per_user_limits_and_they_persist(app, telegram):
    await send(app, msg_update(ADMIN_ID, "/limit 50"))
    assert telegram.texts()[-1] == "✅ Daily limit for everyone: 50."
    await send(app, msg_update(ADMIN_ID, f"/limit {ANA} 200"))
    assert "Ana" in telegram.texts()[-1] and telegram.texts()[-1].endswith(": 200.")
    assert access.global_limit("daily") == 50 and access.limit(ANA, "daily") == (200, False)
    await send(app, msg_update(ADMIN_ID, "/limit"))
    view = telegram.texts()[-1]
    assert "everyone: 50" in view and "Ana" in view and "200" in view
    await send(app, msg_update(ADMIN_ID, f"/limit {ANA} default"))
    assert access.limit(ANA, "daily") == (50, True)


@pytest.mark.parametrize("args", ["abc", "-5", f"{ANA} lots", "1 2 3", f"{ADMIN_ID} 5", "999999 5"])
async def test_bad_arguments_change_nothing(app, telegram, args):
    await send(app, msg_update(ADMIN_ID, f"/limit {args}"))
    assert telegram.texts()[-1].startswith(("Usage:", "⚠️"))
    assert access.global_limit("daily") == 3 and access.limit(ANA, "daily") == (3, True)


async def test_user_sees_own_status_and_cannot_change_it(app, telegram):
    _use(ANA, 1)
    await send(app, msg_update(ANA, "/limit"))
    assert telegram.texts()[-1].split("\n")[0] == "📊 Today: 1 of your 3 requests (last 24 h). 2 left."
    await send(app, msg_update(ANA, "/limit 1000"))
    assert access.global_limit("daily") == 3 and "of your 3 requests" in telegram.texts()[-1]
    _use(ANA, 2)
    await send(app, msg_update(ANA, "/limit"))
    assert "You can send more in about" in telegram.texts()[-1]


async def test_users_list_shows_limits_and_usage(app, telegram):
    _use(ANA, 2)
    db.set_user_limit(BOB, "daily", 200)
    await send(app, msg_update(ADMIN_ID, "/users"))
    texts = "\n".join(telegram.texts())
    assert "no limit" in texts and "Allowed (daily limit: 3)" in texts
    assert "2/3 today (default)" in texts and "0/200 today" in texts


def test_usage_window_and_reset_time():
    _use(ANA, 2)
    count, oldest = db.usage(ANA, "daily")
    assert count == 2 and oldest is not None
    assert db.usage(ANA, "daily", window=0)[0] == 0
    assert render.fmt_until(30) == "1 min" and render.fmt_until(3599) == "60 min" and render.fmt_until(3601) == "2 h"


def test_every_limit_has_a_usage_query_and_a_ui():
    assert set(db.USAGE_FILTERS) == set(access.LIMITS) == set(limits.LIMIT_UI)
