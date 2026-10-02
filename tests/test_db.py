"""Database: users, videos, per-model summaries, requests, migrations."""
import json
import sqlite3
import time

from summarizer import config, db

from conftest import ADMIN_ID


def test_admin_row_exists_after_sync():
    assert db.get_user(ADMIN_ID)["status"] == "admin"


def test_user_lifecycle():
    db.set_user(5, "pending", "Ana", "ana")
    assert db.get_user(5)["status"] == "pending"
    db.set_user(5, "allowed")
    u = db.get_user(5)
    assert (u["status"], u["name"], u["username"]) == ("allowed", "Ana", "ana")  # name kept when not given
    assert [x["id"] for x in db.users_by_status()["allowed"]] == [5]
    db.set_user(5, None)
    assert db.get_user(5) is None


def test_touch_user_updates_known_users_only():
    db.touch_user(ADMIN_ID, "Admin", "adm")
    assert db.get_user(ADMIN_ID)["name"] == "Admin"
    db.touch_user(77, "Stranger", None)
    assert db.get_user(77) is None


def test_llm_choice_round_trip():
    db.set_user_llm(ADMIN_ID, "claude-code", "claude-sonnet-5-5")
    assert db.get_user_llm(ADMIN_ID) == ("claude-code", "claude-sonnet-5-5")
    db.set_user_llm(ADMIN_ID, None, None)
    assert db.get_user_llm(ADMIN_ID) == (None, None)
    assert db.get_user_llm(12345) == (None, None)


def test_sync_admins_removes_former_admins_even_with_empty_list():
    db.sync_admins({ADMIN_ID, 2})
    assert db.get_user(2)["status"] == "admin"
    db.sync_admins({ADMIN_ID})
    assert db.get_user(2) is None
    db.sync_admins(set())
    assert db.get_user(ADMIN_ID) is None


def test_video_and_summaries_per_model():
    db.start_video("youtube", "vid", "https://u")
    db.update_video("youtube", "vid", meta={"title": "T"}, title="T", transcript="hi")
    v = db.get_video("youtube", "vid")
    assert (v["status"], v["meta"], v["transcript"]) == ("processing", {"title": "T"}, "hi")
    db.save_summary("youtube", "vid", "codex", "m1", {"summary": "one"}, False)
    db.save_summary("youtube", "vid", "codex", "m2", {"summary": "two"}, True)
    assert db.get_summary("youtube", "vid", "codex", "m1")["result"]["summary"] == "one"
    s2 = db.get_summary("youtube", "vid", "codex", "m2")
    assert (s2["result"]["summary"], s2["frames_used"]) == ("two", True)
    assert db.get_summary("youtube", "vid", "claude-code", "m1") is None


def test_requests_and_history():
    r1 = db.add_request(5, "https://a", "summary")
    r2 = db.add_request(ADMIN_ID, "https://b", "again")
    db.update_request(r1, platform="youtube", video_id="vid", status="processing")
    assert [r["id"] for r in db.recent_requests(None)] == [r2, r1]  # newest first
    assert [r["id"] for r in db.recent_requests(5)] == [r1]
    db.update_request(r1, status="done")
    row = db.recent_requests(5)[0]
    assert row["status"] == "done" and row["finished_at"] is not None


def test_user_saw_video_ignores_the_current_request():
    r1 = db.add_request(5, "u", "summary")
    db.update_request(r1, platform="youtube", video_id="vid", status="done")
    r2 = db.add_request(5, "u", "summary")
    assert db.user_saw_video(5, "youtube", "vid", except_request=r2)
    assert not db.user_saw_video(5, "youtube", "vid", except_request=r1)
    assert not db.user_saw_video(6, "youtube", "vid", except_request=0)


def test_migration_moves_summaries_out_of_videos(tmp_path, monkeypatch):
    old = tmp_path / "old.sqlite3"
    c = sqlite3.connect(old)
    c.execute("""CREATE TABLE videos (platform TEXT, video_id TEXT, url TEXT, title TEXT, status TEXT, error TEXT,
                 meta TEXT, transcript TEXT, transcript_source TEXT, language TEXT, result TEXT,
                 frames_used INTEGER, created_at REAL, updated_at REAL, PRIMARY KEY (platform, video_id))""")
    c.execute("INSERT INTO videos VALUES ('youtube','v',NULL,'T','done',NULL,'{}','tr','captions','en',?,0,?,?)",
              (json.dumps({"summary": "old"}), time.time(), time.time()))
    c.commit()
    c.close()
    monkeypatch.setattr(db, "PATH", old)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    db.init()
    assert db.get_summary("youtube", "v", "codex", "gpt-6-astra")["result"]["summary"] == "old"
    cols = [r[1] for r in sqlite3.connect(old).execute("PRAGMA table_info(videos)")]
    assert "result" not in cols


def test_stale_requests_are_failed_at_startup():
    queued, running, done = (db.add_request(5, "u", "summary") for _ in range(3))
    db.update_request(running, status="processing")
    db.update_request(done, status="done")
    assert db.fail_stale_requests() == 2
    rows = {r["id"]: r for r in db.recent_requests(5)}
    assert rows[queued]["status"] == rows[running]["status"] == "failed"
    assert rows[queued]["error"] == "bot restarted" and rows[queued]["finished_at"]
    assert rows[done]["status"] == "done"


def test_stats_are_read_once_and_written_atomically(tmp_path, monkeypatch):
    from summarizer import stats
    stats.record("x", 2.0)
    reads = []
    real = type(stats._FILE).read_text
    monkeypatch.setattr(type(stats._FILE), "read_text", lambda self, *a, **k: reads.append(1) or real(self, *a, **k))
    for _ in range(5):
        assert stats.get("x", 0) == 2.0
    assert reads == [] and not stats._FILE.with_name("stats.json.tmp").exists()


def test_database_uses_wal():
    with db._db() as c:
        assert c.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
