"""The bot's database (data/bot.sqlite3).

users     everyone: admins (synced from .env) and allowed / pending / blocked users, with their AI choice
videos    one row per video (platform, video_id): metadata and transcript, as they come in
summaries one row per video and LLM (backend + model): each user gets the summary of the model they use
requests  one row per link a user sends: who, what, when, how it went

Who submitted which video is private: only admins can see other users' requests.

All access goes through _db(): one process-wide lock plus a short-lived connection per call. The bot's
event loop and the pipeline's worker thread both write, and sqlite3 connections must not be shared
between threads.
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
    created_at        REAL NOT NULL,
    updated_at        REAL NOT NULL,
    PRIMARY KEY (platform, video_id)
);
CREATE TABLE IF NOT EXISTS summaries (
    platform     TEXT NOT NULL,
    video_id     TEXT NOT NULL,
    backend      TEXT NOT NULL,                -- codex | claude-code | api
    model        TEXT NOT NULL,                -- the model requested (default resolved), e.g. gpt-6-sol
    result       TEXT NOT NULL,                -- JSON summary (+ _stats)
    frames_used  INTEGER DEFAULT 0,
    created_at   REAL NOT NULL,
    PRIMARY KEY (platform, video_id, backend, model)
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
    """Open a connection that returns rows as sqlite3.Row.

    Returns:
        A new connection to the bot's database.
    """
    # timeout: wait for a lock rather than fail if another process (e.g. a manual sqlite3 session) holds it.
    c = sqlite3.connect(PATH, timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")  # readers don't block the writer (and vice versa)
    return c


@contextmanager
def _db():
    """Run one transaction on a fresh connection, serialized across threads.

    Yields:
        The connection; changes are committed on success and rolled back on an exception.
    """
    with _lock:
        c = _connect()
        try:
            with c:  # commit / rollback
                yield c
        finally:
            c.close()


def init() -> None:
    """Create the tables and migrate databases created by older versions of the bot.

    Runs at import, so every entry point (bot, scripts, tests) sees the current schema. Migrations must
    keep existing rows: ALTER/move data, never drop a table that still holds data.
    """
    with _db() as c:
        c.executescript(SCHEMA)
        cols = {r["name"] for r in c.execute("PRAGMA table_info(users)")}
        for col in ("backend", "model"):  # databases created before per-user models
            if col not in cols:
                c.execute(f"ALTER TABLE users ADD COLUMN {col} TEXT")
        vcols = {r["name"] for r in c.execute("PRAGMA table_info(videos)")}
        if "result" in vcols:  # summaries used to live in videos (one per video): move them out
            for r in c.execute("SELECT platform, video_id, result, frames_used, updated_at FROM videos "
                               "WHERE result IS NOT NULL").fetchall():
                st = json.loads(r["result"]).get("_stats", {})
                # Summaries from before model tracking were written by Codex's default then, gpt-6-astra.
                c.execute("INSERT OR IGNORE INTO summaries VALUES (?,?,?,?,?,?,?)",
                          (r["platform"], r["video_id"], st.get("backend") or "codex",
                           st.get("model") or "gpt-6-astra", r["result"], r["frames_used"] or 0, r["updated_at"]))
            c.execute("ALTER TABLE videos DROP COLUMN result")
            c.execute("ALTER TABLE videos DROP COLUMN frames_used")
        if c.execute("SELECT 1 FROM sqlite_master WHERE name='user_settings'").fetchone():
            for r in c.execute("SELECT user_id, model FROM user_settings").fetchall():
                c.execute("UPDATE users SET model=? WHERE id=?", (r["model"], r["user_id"]))
            c.execute("DROP TABLE user_settings")
    _migrate_old_files()


def sync_admins(admin_ids: set[int]) -> None:
    """Give the .env admins a users row (status 'admin') so their settings live there too.

    .env stays the source of truth for who is admin (nobody can lock themselves out from Telegram).
    Someone removed from ADMIN_USER_IDS loses the admin row, and their settings with it.

    Args:
        admin_ids: The admin user ids from ADMIN_USER_IDS.
    """
    now = time.time()
    with _db() as c:
        for uid in admin_ids:
            c.execute("""INSERT INTO users (id, status, created_at, updated_at) VALUES (?, 'admin', ?, ?)
                         ON CONFLICT(id) DO UPDATE SET status='admin'""", (uid, now, now))
        if not admin_ids:
            # Setup mode: nobody is admin. (Can't use the query below: "NOT IN ()" is a syntax error, and
            # "NOT IN (NULL)" is never true, so it would keep former admins' rows.)
            c.execute("DELETE FROM users WHERE status='admin'")
            return
        placeholders = ",".join("?" * len(admin_ids))
        c.execute(f"DELETE FROM users WHERE status='admin' AND id NOT IN ({placeholders})", tuple(admin_ids))


# ---------- users ----------

def get_user(uid: int) -> dict | None:
    """Look up a user.

    Args:
        uid: Telegram user id.

    Returns:
        The users row as a dict, or None if the bot has never seen them.
    """
    with _db() as c:
        row = c.execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone()
    return dict(row) if row else None


def set_user(uid: int, status: str | None, name: str | None = None, username: str | None = None) -> dict | None:
    """Create or update a user, or delete them.

    Deleting also drops their settings (AI choice): a removed user who comes back starts fresh.

    Args:
        uid: Telegram user id.
        status: "allowed", "pending" or "blocked"; None deletes the user.
        name: Display name; None keeps the stored one.
        username: Telegram @username without the @; None keeps the stored one.

    Returns:
        The updated row, or for a delete the row as it was (None if the user was unknown).
    """
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


def touch_user(uid: int, name: str | None, username: str | None) -> None:
    """Keep an existing user's name and username current (Telegram names change).

    Called only on /start and on an admin's /users, not on every message, to keep writes rare.
    Does nothing for users without a row.

    Args:
        uid: Telegram user id.
        name: Current display name.
        username: Current @username without the @, or None.
    """
    with _db() as c:
        # IS NOT (unlike !=) treats NULLs as comparable, so a missing username still counts as a change.
        c.execute("UPDATE users SET name=?, username=? WHERE id=? AND (name IS NOT ? OR username IS NOT ?)",
                  (name, username, uid, name, username))


def users_by_status() -> dict[str, list[dict]]:
    """All users grouped by status, oldest first.

    Returns:
        {"admin": [...], "allowed": [...], "pending": [...], "blocked": [...]}; every key is present.
    """
    out = {"admin": [], "allowed": [], "pending": [], "blocked": []}
    with _db() as c:
        for row in c.execute("SELECT * FROM users ORDER BY created_at"):
            out.setdefault(row["status"], []).append(dict(row))
    return out


def get_user_llm(uid: int) -> tuple[str | None, str | None]:
    """The AI a user chose in /models.

    Args:
        uid: Telegram user id.

    Returns:
        (backend, model); either is None when the user didn't choose (= the default).
    """
    with _db() as c:
        row = c.execute("SELECT backend, model FROM users WHERE id=?", (uid,)).fetchone()
    return (row["backend"], row["model"]) if row else (None, None)


def set_user_llm(uid: int, backend: str | None, model: str | None) -> None:
    """Store a user's AI choice.

    Args:
        uid: Telegram user id; the user must already have a row.
        backend: "codex", "claude-code" or "api"; None = default.
        model: A model of that backend; None = its default. (None, None) = back to the defaults.
    """
    with _db() as c:
        c.execute("UPDATE users SET backend=?, model=?, updated_at=? WHERE id=?",
                  (backend, model, time.time(), uid))


# ---------- videos ----------

def get_video(platform: str, video_id: str) -> dict | None:
    """Look up a video's metadata and transcript.

    Args:
        platform: "youtube" or "tiktok".
        video_id: The platform's id.

    Returns:
        The videos row with "meta" decoded from JSON, or None if the video was never processed.
    """
    with _db() as c:
        row = c.execute("SELECT * FROM videos WHERE platform=? AND video_id=?", (platform, video_id)).fetchone()
    if not row:
        return None
    v = dict(row)
    v["meta"] = json.loads(v["meta"]) if v["meta"] else None
    return v


def start_video(platform: str, video_id: str, url: str) -> None:
    """Create the video's row (status "processing") as soon as work starts.

    An existing row only gets its status reset: the saved transcript and metadata stay, so /again and
    other models' summaries reuse them instead of downloading and transcribing again.

    Args:
        platform: "youtube" or "tiktok".
        video_id: The platform's id.
        url: Canonical URL.
    """
    now = time.time()
    with _db() as c:
        c.execute("""INSERT INTO videos (platform, video_id, url, status, created_at, updated_at)
                     VALUES (?,?,?, 'processing', ?, ?)
                     ON CONFLICT(platform, video_id) DO UPDATE SET status='processing', error=NULL,
                       updated_at=excluded.updated_at""",
                  (platform, video_id, url, now, now))


def update_video(platform: str, video_id: str, **fields) -> None:
    """Fill in a video's columns as data arrives.

    Args:
        platform: "youtube" or "tiktok".
        video_id: The platform's id.
        **fields: Column values, e.g. title=..., transcript=..., status=...; "meta" is JSON-encoded here.
            Column names come from code, never from user input (they're formatted into the SQL).
    """
    if fields.get("meta") is not None:
        fields["meta"] = json.dumps(fields["meta"], ensure_ascii=False)
    fields["updated_at"] = time.time()
    cols = ", ".join(f"{k}=?" for k in fields)
    with _db() as c:
        c.execute(f"UPDATE videos SET {cols} WHERE platform=? AND video_id=?",
                  (*fields.values(), platform, video_id))


# ---------- summaries ----------

def get_summary(platform: str, video_id: str, backend: str, model: str) -> dict | None:
    """Look up the summary a given model wrote for a video.

    Args:
        platform: "youtube" or "tiktok".
        video_id: The platform's id.
        backend: "codex", "claude-code" or "api".
        model: The model id (the default already resolved).

    Returns:
        {"result": {...}, "frames_used": bool}, or None if this model hasn't summarized the video.
    """
    with _db() as c:
        row = c.execute("SELECT result, frames_used FROM summaries WHERE platform=? AND video_id=? "
                        "AND backend=? AND model=?", (platform, video_id, backend, model)).fetchone()
    return {"result": json.loads(row["result"]), "frames_used": bool(row["frames_used"])} if row else None


def save_summary(platform: str, video_id: str, backend: str, model: str, result: dict, frames_used: bool) -> None:
    """Store a model's summary of a video, replacing that model's earlier one (/again).

    Keyed by backend + model, so users on different models never overwrite each other's summaries.

    Args:
        platform: "youtube" or "tiktok".
        video_id: The platform's id.
        backend: "codex", "claude-code" or "api".
        model: The model id (the default already resolved).
        result: The LLM's answer plus "_stats".
        frames_used: Whether frames or slides were shown to the LLM.
    """
    with _db() as c:
        c.execute("INSERT OR REPLACE INTO summaries VALUES (?,?,?,?,?,?,?)",
                  (platform, video_id, backend, model, json.dumps(result, ensure_ascii=False),
                   int(frames_used), time.time()))


# ---------- requests ----------

def add_request(user_id: int, url: str, kind: str) -> int:
    """Log a link the moment it arrives (status "queued"), before any processing.

    Logging first means even links that fail early (bad URL, crash) appear in /history.

    Args:
        user_id: Telegram user id of the sender.
        url: The link as sent.
        kind: "summary", "again" or "transcript".

    Returns:
        The new request id.
    """
    with _db() as c:
        cur = c.execute("INSERT INTO requests (user_id, url, kind, status, created_at) VALUES (?,?,?,'queued',?)",
                        (user_id, url, kind, time.time()))
        return cur.lastrowid


def update_request(req_id: int, **fields) -> None:
    """Update a request as it moves through the queue.

    Args:
        req_id: The request id from add_request().
        **fields: Column values, e.g. status=..., platform=..., cached=..., error=...; finished_at is set
            automatically when the status becomes "done" or "failed". Column names come from code only.
    """
    if fields.get("status") in ("done", "failed", "cancelled"):
        fields["finished_at"] = time.time()
    cols = ", ".join(f"{k}=?" for k in fields)
    with _db() as c:
        c.execute(f"UPDATE requests SET {cols} WHERE id=?", (*fields.values(), req_id))


def fail_stale_requests() -> int:
    """Marks requests left unfinished by a crash or kill as failed ("bot restarted"); call at startup.

    The queue lives in memory, so after a restart nothing will ever finish them; without this they'd show
    as running in /history forever.

    Returns:
        How many requests were marked.
    """
    with _db() as c:
        cur = c.execute("""UPDATE requests SET status='failed', error='bot restarted', finished_at=?
                           WHERE status IN ('queued', 'processing')""", (time.time(),))
        return cur.rowcount


def last_again(user_id: int, platform: str, video_id: str, except_request: int) -> float | None:
    """When this user last successfully redid (/again) this video, or None if never.

    Args:
        user_id: Telegram user id.
        platform: "youtube" or "tiktok".
        video_id: The video's id on that platform.
        except_request: The current request, which mustn't count itself.

    Returns:
        The unix time it finished, or None.
    """
    with _db() as c:
        row = c.execute("""SELECT MAX(finished_at) AS t FROM requests WHERE user_id=? AND platform=? AND video_id=?
                           AND kind='again' AND status='done' AND id<>?""",
                        (user_id, platform, video_id, except_request)).fetchone()
    return row["t"] if row else None


def user_saw_video(user_id: int, platform: str, video_id: str, except_request: int) -> bool:
    """Whether this user successfully requested this video before.

    Only then may they be told a result came from the cache; otherwise "from cache" would reveal that
    someone else submitted the video.

    Args:
        user_id: Telegram user id.
        platform: "youtube" or "tiktok".
        video_id: The platform's id.
        except_request: The current request's id, which mustn't count as "before".

    Returns:
        True if an earlier request of theirs for this video finished.
    """
    with _db() as c:
        row = c.execute("""SELECT 1 FROM requests WHERE user_id=? AND platform=? AND video_id=?
                           AND status='done' AND id<>? LIMIT 1""",
                        (user_id, platform, video_id, except_request)).fetchone()
    return row is not None


def recent_requests(user_id: int | None, limit: int = 15) -> list[dict]:
    """Recent requests, newest first, with the video title and sender's name.

    Args:
        user_id: Only this user's requests; None = everyone's (admins only, it shows who sent what).
        limit: Maximum number of rows.

    Returns:
        Request rows as dicts, plus "title", "user_name" and "user_username".
    """
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
    """Import data from the bot's earlier storage files, then delete them.

    cache.sqlite3 held videos with one summary each; users.json held the allowed/pending/blocked users.
    Runs on every start and does nothing once the files are gone.
    """
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
                             transcript_source, language, created_at, updated_at)
                             VALUES (?,?,?,?,?,?,?,?,?,?)""",
                          (p, vid, title, "done" if res else "failed", meta, tr, ts, lang, created, created))
                if res:
                    # The old cache predates per-model summaries; Codex's default then was gpt-6-astra.
                    c.execute("INSERT OR IGNORE INTO summaries VALUES (?,?,?,?,?,?,?)",
                              (p, vid, "codex", "gpt-6-astra", res, fu or 0, created))
        for f in config.DATA_DIR.glob("cache.sqlite3*"):  # also its -wal / -shm files
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


init()  # at import: every user of this module needs the schema in place first
