"""The bot's database (data/bot.sqlite3).

users     people other than the .env admins: allowed / pending / blocked
videos    one row per video (platform, video_id): metadata, transcript, summary, as they come in
requests  one row per link a user sends: who, what, when, how it went

Who submitted which video is private: only admins can see other users' requests.
"""
import json
import sqlite3
import threading
import time
from contextlib import contextmanager

from . import config

PATH = config.DATA_DIR / "bot.sqlite3"
_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY,           -- Telegram user id
    name        TEXT,
    username    TEXT,
    status      TEXT NOT NULL,                 -- admin (from .env) | allowed | pending | blocked
    backend     TEXT,                          -- chosen LLM backend (codex | claude-code | api); NULL = default
    model       TEXT,                          -- chosen model of that backend; NULL = its default
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS videos (
    platform          TEXT NOT NULL,
    video_id          TEXT NOT NULL,
    url               TEXT,
    title             TEXT,
    status            TEXT NOT NULL,           -- processing | done | failed
    error             TEXT,
    meta              TEXT,                    -- JSON
    transcript        TEXT,
    transcript_source TEXT,
    language          TEXT,
    result            TEXT,                    -- JSON summary (+ _stats)
    frames_used       INTEGER DEFAULT 0,
    created_at        REAL NOT NULL,
    updated_at        REAL NOT NULL,
    PRIMARY KEY (platform, video_id)
);
CREATE TABLE IF NOT EXISTS requests (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL,
    url         TEXT NOT NULL,
    kind        TEXT NOT NULL,                 -- summary | again | transcript
    platform    TEXT,
    video_id    TEXT,
    status      TEXT NOT NULL,                 -- queued | processing | done | failed
    cached      INTEGER DEFAULT 0,
    error       TEXT,
    created_at  REAL NOT NULL,
    finished_at REAL
);
CREATE INDEX IF NOT EXISTS requests_user ON requests (user_id, created_at);
CREATE INDEX IF NOT EXISTS requests_video ON requests (platform, video_id);
"""


def _connect() -> sqlite3.Connection:
    c = sqlite3.connect(PATH, timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    return c


@contextmanager
def _db():
    with _lock:
        c = _connect()
        try:
            with c:  # commit / rollback
                yield c
        finally:
            c.close()


def init() -> None:
    with _db() as c:
        c.executescript(SCHEMA)
        cols = {r["name"] for r in c.execute("PRAGMA table_info(users)")}
        for col in ("backend", "model"):  # databases created before per-user models
            if col not in cols:
                c.execute(f"ALTER TABLE users ADD COLUMN {col} TEXT")
        if c.execute("SELECT 1 FROM sqlite_master WHERE name='user_settings'").fetchone():
            for r in c.execute("SELECT user_id, model FROM user_settings").fetchall():
                c.execute("UPDATE users SET model=? WHERE id=?", (r["model"], r["user_id"]))
            c.execute("DROP TABLE user_settings")
    _migrate_old_files()


def sync_admins(admin_ids: set[int]) -> None:
    """Admins come from .env; give them a users row too (status 'admin') so their settings live there.
    Someone removed from ADMIN_USER_IDS loses the admin row (their settings go with it)."""
    now = time.time()
    with _db() as c:
        for uid in admin_ids:
            c.execute("""INSERT INTO users (id, status, created_at, updated_at) VALUES (?, 'admin', ?, ?)
                         ON CONFLICT(id) DO UPDATE SET status='admin'""", (uid, now, now))
        placeholders = ",".join("?" * len(admin_ids)) or "NULL"
        c.execute(f"DELETE FROM users WHERE status='admin' AND id NOT IN ({placeholders})", tuple(admin_ids))


# ---------- users ----------

def get_user(uid: int) -> dict | None:
    with _db() as c:
        row = c.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    return dict(row) if row else None


def set_user(uid: int, status: str | None, name: str | None = None, username: str | None = None) -> dict | None:
    """Create/update a user; status None deletes. Returns the (previous) row, or None if unknown."""
    now = time.time()
    with _db() as c:
        old = c.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
        old = dict(old) if old else None
        if status is None:
            c.execute("DELETE FROM users WHERE id=?", (uid,))
            return old
        c.execute("""INSERT INTO users (id, name, username, status, created_at, updated_at)
                     VALUES (?,?,?,?,?,?)
                     ON CONFLICT(id) DO UPDATE SET status=excluded.status, updated_at=excluded.updated_at,
                       name=COALESCE(excluded.name, name), username=COALESCE(excluded.username, username)""",
                  (uid, name, username, status, now, now))
        row = c.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    return dict(row)


def users_by_status() -> dict[str, list[dict]]:
    out = {"admin": [], "allowed": [], "pending": [], "blocked": []}
    with _db() as c:
        for row in c.execute("SELECT * FROM users ORDER BY created_at"):
            out.setdefault(row["status"], []).append(dict(row))
    return out


def get_user_llm(uid: int) -> tuple[str | None, str | None]:
    """(backend, model) the user chose; None = default."""
    with _db() as c:
        row = c.execute("SELECT backend, model FROM users WHERE id=?", (uid,)).fetchone()
    return (row["backend"], row["model"]) if row else (None, None)


def set_user_llm(uid: int, backend: str | None, model: str | None) -> None:
    """(None, None) = back to the defaults."""
    with _db() as c:
        c.execute("UPDATE users SET backend=?, model=?, updated_at=? WHERE id=?",
                  (backend, model, time.time(), uid))


# ---------- videos ----------

def get_video(platform: str, video_id: str) -> dict | None:
    with _db() as c:
        row = c.execute("SELECT * FROM videos WHERE platform=? AND video_id=?", (platform, video_id)).fetchone()
    if not row:
        return None
    v = dict(row)
    v["meta"] = json.loads(v["meta"]) if v["meta"] else None
    v["result"] = json.loads(v["result"]) if v["result"] else None
    v["frames_used"] = bool(v["frames_used"])
    return v


def start_video(platform: str, video_id: str, url: str) -> None:
    """Row exists as soon as work starts; keeps earlier transcript/result for /again."""
    now = time.time()
    with _db() as c:
        c.execute("""INSERT INTO videos (platform, video_id, url, status, created_at, updated_at)
                     VALUES (?,?,?, 'processing', ?, ?)
                     ON CONFLICT(platform, video_id) DO UPDATE SET status='processing', error=NULL,
                       updated_at=excluded.updated_at""",
                  (platform, video_id, url, now, now))


def update_video(platform: str, video_id: str, **fields) -> None:
    """Fill in columns as data arrives (meta/result are JSON-encoded here)."""
    for k in ("meta", "result"):
        if k in fields and fields[k] is not None:
            fields[k] = json.dumps(fields[k], ensure_ascii=False)
    if "frames_used" in fields:
        fields["frames_used"] = int(bool(fields["frames_used"]))
    fields["updated_at"] = time.time()
    cols = ", ".join(f"{k}=?" for k in fields)
    with _db() as c:
        c.execute(f"UPDATE videos SET {cols} WHERE platform=? AND video_id=?",
                  (*fields.values(), platform, video_id))


# ---------- requests ----------

def add_request(user_id: int, url: str, kind: str) -> int:
    with _db() as c:
        cur = c.execute("INSERT INTO requests (user_id, url, kind, status, created_at) VALUES (?,?,?,'queued',?)",
                        (user_id, url, kind, time.time()))
        return cur.lastrowid


def update_request(req_id: int, **fields) -> None:
    if fields.get("status") in ("done", "failed"):
        fields["finished_at"] = time.time()
    cols = ", ".join(f"{k}=?" for k in fields)
    with _db() as c:
        c.execute(f"UPDATE requests SET {cols} WHERE id=?", (*fields.values(), req_id))


def user_saw_video(user_id: int, platform: str, video_id: str, except_request: int) -> bool:
    """Has this user successfully requested this video before? (Only then may they see it was cached.)"""
    with _db() as c:
        row = c.execute("""SELECT 1 FROM requests WHERE user_id=? AND platform=? AND video_id=?
                           AND status='done' AND id<>? LIMIT 1""",
                        (user_id, platform, video_id, except_request)).fetchone()
    return row is not None


def recent_requests(user_id: int | None, limit: int = 15) -> list[dict]:
    """Newest first, with the video title. user_id None = everyone's (admins only)."""
    q = """SELECT r.*, v.title, u.name AS user_name, u.username AS user_username
           FROM requests r
           LEFT JOIN videos v ON v.platform=r.platform AND v.video_id=r.video_id
           LEFT JOIN users u ON u.id=r.user_id"""
    args: tuple = ()
    if user_id is not None:
        q += " WHERE r.user_id=?"
        args = (user_id,)
    q += " ORDER BY r.id DESC LIMIT ?"
    with _db() as c:
        return [dict(r) for r in c.execute(q, (*args, limit))]


# ---------- one-time import of the old cache.sqlite3 / users.json ----------

def _migrate_old_files() -> None:
    old_cache = config.DATA_DIR / "cache.sqlite3"
    if old_cache.exists():
        src = sqlite3.connect(old_cache)
        rows = src.execute("SELECT platform, video_id, meta, transcript, transcript_source, language, result, "
                           "frames_used, created FROM videos").fetchall()
        src.close()
        with _db() as c:
            for p, vid, meta, tr, ts, lang, res, fu, created in rows:
                title = (json.loads(meta) or {}).get("title") if meta else None
                c.execute("""INSERT OR IGNORE INTO videos (platform, video_id, title, status, meta, transcript,
                             transcript_source, language, result, frames_used, created_at, updated_at)
                             VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                          (p, vid, title, "done" if res else "failed", meta, tr, ts, lang, res, fu,
                           created, created))
        for f in config.DATA_DIR.glob("cache.sqlite3*"):
            f.unlink()
    old_users = config.DATA_DIR / "users.json"
    if old_users.exists():
        data = json.loads(old_users.read_text() or "{}")
        with _db() as c:
            for status, people in data.items():
                for uid, info in people.items():
                    c.execute("""INSERT OR IGNORE INTO users (id, name, username, status, created_at, updated_at)
                                 VALUES (?,?,?,?,?,?)""",
                              (int(uid), info.get("name"), info.get("username"), status,
                               info.get("at", time.time()), info.get("at", time.time())))
        old_users.unlink()


init()
