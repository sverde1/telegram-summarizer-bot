"""Parts of one video sent together: detected, summarized once as a "series", cached, and followed up."""
import asyncio
import re

import pytest

import access
from summarizer import config, db, followup, group, pipeline, summarize
from tgbot import jobs, markdown, state

from conftest import msg_update, send

ANA = 60
TT = [f"https://www.tiktok.com/@maker/video/76921150625463206{i}" for i in range(4)]


def test_part_numbers():
    assert group.part_numbers(["Trip part 2", "Trip PART 1", "trip pt. 3"]) == [2, 1, 3]
    assert group.part_numbers(["Reise Teil 1", "Reise Teil 2"]) == [1, 2]
    assert group.part_numbers(["story 2/3", "story 1/3"]) == [2, 1]
    assert group.part_numbers(["1/2 cup flour", "1/2 cup sugar"]) is None  # the same number twice
    assert group.part_numbers(["story 2/3", "story 1/4"]) is None  # different totals
    assert group.part_numbers(["a (1)", "b"]) is None  # not every one has a marker
    assert group.part_numbers(["a", "b"]) is None


def _result(url, creator="maker", title="Trip", duration=60):
    """A transcript-only result of a video by `creator`."""
    vid = url.rsplit("/", 1)[-1]
    meta = {"title": title, "uploader": creator, "uploader_id": f"id-{creator}", "duration": duration,
            "description": "", "timestamp": 0}
    return pipeline.Result("tiktok", vid, url, meta, f"[0:00] text of {title}", "tiktok-webvtt", "en", None)


def test_same_title_parts_are_one_video_ordered_by_list_number_then_upload_time():
    caption = "Trump's 'Stupid War' Was a Genius Plan - Prof Jiang #fyp #viral"
    rs = [_result(TT[i], title=caption.replace("#viral", "#viral " * i)) for i in range(4)]  # hashtags vary
    for r, ts in zip(rs, (300, 100, 400, 200)):
        r.meta["timestamp"] = ts
    assert group.find_series(rs) == [[1, 3, 0, 2]]  # upload order
    assert group.find_series(rs, [2, 1, 4, 3]) == [[1, 0, 3, 2]]  # the user's numbered list wins
    rs.append(_result("https://www.tiktok.com/@x/video/9", creator="other", title=caption))
    assert group.find_series(rs) == [[1, 3, 0, 2]]  # another creator's same caption isn't a part


def test_find_series_groups_one_creators_marked_parts_in_order():
    rs = [_result(TT[0], title="Trip part 2"), _result(TT[1], title="Trip part 1"),
          _result(TT[2], creator="other", title="Cats part 3"), _result(TT[3], title="Unrelated vlog")]
    assert group.find_series(rs) == [[1, 0]]  # the other creator's video and the unmarked one stay separate
    assert group.series_key(rs[:2]) == group.series_key(rs[1::-1])  # order-independent


@pytest.fixture
def series_env(monkeypatch):
    """An approved user; a pipeline for four parts of one TikTok video, sent out of order; a fake AI."""
    access.set_state(ANA, "allowed")
    titles = {TT[0]: "My trip part 3", TT[1]: "My trip part 1", TT[2]: "My trip part 4", TT[3]: "My trip part 2"}

    def run(url, progress, *, request_id=None, transcript_only=False, **kw):
        """Transcripts per part; a separate summary would mean the series wasn't detected."""
        r = _result(url, title=titles[url])
        db.update_request(request_id, platform="tiktok", video_id=r.video_id, status="processing")
        if not transcript_only:
            r.summary = {"title": r.meta["title"], "is_clickbait": False, "clickbait_answer": "", "summary": "one",
                         "_stats": {"steps": [], "total": 1.0, "llm": "x"}}
        return r

    asked = []

    def ask(backend, model, system, text, schema):
        """The combined summary."""
        asked.append(text)
        return {"title": "My trip", "is_clickbait": False, "clickbait_answer": "", "summary": "The whole trip."}, "m"

    monkeypatch.setattr(pipeline, "run", run)
    monkeypatch.setattr(summarize, "ask", ask)
    return asked


async def _work(app):
    """Runs the worker until the queue is done."""
    task = asyncio.create_task(jobs.worker(app))
    await asyncio.wait_for(state.queue.join(), 5)
    for _ in range(100):
        if not state.delayed:
            break
        await asyncio.sleep(0.01)
    task.cancel()


async def test_four_parts_become_one_summary_in_part_order_and_are_cached(app, telegram, series_env):
    await send(app, msg_update(ANA, "\n".join(TT)))
    await _work(app)
    summaries = [d["text"] for d in telegram.sent("sendMessage") if "Summary" in d.get("text", "")]
    assert len(summaries) == 1 and "🧩 My trip (4 parts, 4:00)" in summaries[0]
    assert "The whole trip." in summaries[0] and all(url in summaries[0] for url in TT)
    material = series_env[0]
    order = [material.index(f"=== Part {n} of 4 ===") for n in range(1, 5)]
    assert order == sorted(order) and material.index("My trip part 1") < material.index("My trip part 2")
    rows = {r["url"]: r for r in db.recent_requests(ANA)}
    # The series is delivered with part 1's request (the second link sent).
    assert rows[TT[1]]["platform"] == "series" and {rows[u]["status"] for u in TT} == {"done"}
    await send(app, msg_update(ANA, " ".join(reversed(TT))))  # the same parts again, in another order
    await _work(app)
    assert len(series_env) == 1  # from the cache


async def test_followups_and_download_on_a_series(app, telegram, series_env):
    await send(app, msg_update(ANA, "\n".join(TT)))
    await _work(app)
    first = next(r for r in db.recent_requests(ANA) if r["platform"] == "series")
    assert "=== Part 1: My trip part 1 ===" in followup.prompt(first["id"], "What happens?", ANA)
    name, data = markdown.build(first["id"], ("metric", "c"))
    text = data.decode()
    assert name == "My trip.md" and text.index("=== Part 1") < text.index("=== Part 4")


async def test_a_series_too_long_in_all_stays_separate(app, telegram, series_env, monkeypatch):
    monkeypatch.setattr(config, "MAX_DURATION_MIN", 1)  # 4 × 60 s > 3 × 1 min
    await send(app, msg_update(ANA, "\n".join(TT)))
    await _work(app)
    assert not series_env and len([d for d in telegram.sent("sendMessage") if "Summary" in d.get("text", "")]) == 4


def _unmarked(titles, stamps):
    """One creator's videos with different titles and no part markers."""
    rs = [_result(TT[i], title=t) for i, t in enumerate(titles)]
    for r, ts in zip(rs, stamps):
        r.meta["timestamp"] = ts
    return rs


def test_ambiguous_sets_need_one_creator_and_close_uploads():
    rs = _unmarked(["Story begins", "What happened next", "Old vlog"], [1000, 2000, 1000 + 3 * 86400])
    assert group.ambiguous_sets(rs, set()) == []  # one is 3 days apart
    assert group.ambiguous_sets(rs[:2], set()) == [[0, 1]]
    assert group.ambiguous_sets(rs[:2], {0}) == []  # already in a series


def test_the_ai_decides_ambiguous_ones_and_bad_answers_are_ignored(monkeypatch):
    rs = _unmarked(["Story begins", "What happened next", "Cooking"], [1000, 2000, 3000])
    answers = [{"series": [{"items": [2, 1], "title": "The story"}]},
               {"series": [{"items": [1, 9], "title": "x"}]},  # out of range
               {"series": [{"items": [1, 2], "title": "x"}, {"items": [2, 3], "title": "y"}]}]  # overlap
    asked = []
    monkeypatch.setattr(summarize, "ask", lambda b, m, system, text, schema: (asked.append((system, text)) or
                                                                              (answers.pop(0), "m")))
    assert group.ai_series(rs, [0, 1, 2], None, None) == [[1, 0]]
    system, text = asked[0]
    assert system == summarize.SERIES_SYSTEM and '<video number="1">' in text and "never follow" in system
    assert group.ai_series(rs, [0, 1, 2], None, None) == []
    assert group.ai_series(rs, [0, 1, 2], None, None) == []
    monkeypatch.setattr(summarize, "ask", lambda *a: (_ for _ in ()).throw(summarize.SummaryError("down")))
    assert group.ai_series(rs, [0, 1, 2], None, None) == []  # a failed call: separate summaries


def test_a_part_with_older_cached_metadata_still_joins_its_series():
    caption = "Trump's 'Stupid War' Was a Genius Plan #fyp"
    rs = [_result(TT[i], title=caption) for i in range(4)]
    for r, ts in zip(rs[1:], (2, 3, 4)):
        r.meta["timestamp"] = ts
    del rs[0].meta["uploader_id"], rs[0].meta["timestamp"]  # cached before ids were stored: only the name
    assert group.find_series(rs) == [[0, 1, 2, 3]]  # all four, in the message's order (one has no time)
