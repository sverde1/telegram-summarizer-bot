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
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,                    -- e.g. daily_limit (set with /limit)
    value TEXT NOT NULL
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
CREATE TABLE IF NOT EXISTS uploads (
    id          INTEGER PRIMARY KEY AUTOINCREMENT, -- used in the upload's buttons (book:<id>:...)
    user_id     INTEGER NOT NULL,
    file_id     TEXT NOT NULL,                 -- Telegram's id to download the file again ("" for links)
    file_unique_id TEXT,
    name        TEXT NOT NULL,                 -- file name as sent (shown to the user and in /history)
    size        INTEGER,
    sha256      TEXT,                          -- set once downloaded; the documents row
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS documents (
    sha256      TEXT PRIMARY KEY,              -- of the file's bytes: identical uploads share the work
    name        TEXT,
    format      TEXT,                          -- pdf | epub | docx | txt
    pages       INTEGER,                       -- real pages (PDF) or ~2000-character pages
    title       TEXT,                          -- from the file's metadata, may be empty
    author      TEXT,
    language    TEXT,                          -- OCR language, when OCR was used
    text_source TEXT,                          -- text | ocr-tesseract | ocr-rapidocr
    chapters    TEXT,                          -- JSON [{title, start, end}], page indices, end exclusive
    status      TEXT NOT NULL,                 -- processing | done | failed
    error       TEXT,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS document_pages (
    sha256  TEXT NOT NULL,
    page    INTEGER NOT NULL,                  -- 0-based
    text    TEXT NOT NULL,
    source  TEXT NOT NULL,                     -- text | ocr-<engine>; OCR pages are saved as they finish
    PRIMARY KEY (sha256, page)
);
CREATE TABLE IF NOT EXISTS doc_summaries (
    sha256   TEXT NOT NULL,
    part     TEXT NOT NULL,                    -- "book", or a chapter index
    style    TEXT NOT NULL,                    -- short | full (chapters); full (book)
    backend  TEXT NOT NULL,
    model    TEXT NOT NULL,
    result   TEXT NOT NULL,                    -- JSON: {summary} for chapters, {title, author, summary, _stats}
    created_at REAL NOT NULL,
    PRIMARY KEY (sha256, part, style, backend, model)
);
CREATE TABLE IF NOT EXISTS ocr_holds (
    request_id  INTEGER PRIMARY KEY,           -- a document request waiting for the user to confirm OCR
    upload_id   INTEGER NOT NULL,
    mode        TEXT NOT NULL,                 -- the job to continue: whole | short | each | pick
    chapter     INTEGER,
    pages       INTEGER NOT NULL,              -- pages that need OCR
    seconds     REAL NOT NULL,                 -- estimated OCR time
    language    TEXT,                          -- OCR language(s), e.g. "slv"
    asked       INTEGER DEFAULT 0,             -- the user asked the admins to allow a long OCR
    approved    INTEGER DEFAULT 0,             -- an admin allowed it (over OCR_MAX_PAGES)
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS spoken (
    request_id  INTEGER PRIMARY KEY,           -- the summary request whose 🔊 button this is
    title       TEXT NOT NULL,
    text        TEXT NOT NULL,                 -- exactly what is read aloud (what the user read)
    lang        TEXT NOT NULL,                 -- espeak language code and Kokoro voice, as when delivered
    voice       TEXT NOT NULL,
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS voices (
    key         TEXT PRIMARY KEY,              -- sha256 of voice, language, speed and text (no user)
    file_id     TEXT NOT NULL,                 -- Telegram's id of the sent voice message, reusable in any chat
    duration    INTEGER,
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS delivered (
    request_id  INTEGER PRIMARY KEY,           -- a summary (or an answer) the user received
    parent_id   INTEGER,                       -- for an answer: the summary request it was about
    kind        TEXT NOT NULL,                 -- video | file | document | ask
    title       TEXT NOT NULL,                 -- for an answer: the question
    text        TEXT NOT NULL,                 -- what was shown, before unit conversion
    backend     TEXT,                          -- the resolved backend and model that wrote it
    model       TEXT,
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    chat_id     INTEGER NOT NULL,              -- a bot message a user can reply to with a question:
    message_id  INTEGER NOT NULL,              -- every part of a summary or answer, and the 💬 prompt
    request_id  INTEGER NOT NULL,              -- the summary request a reply to it asks about
    PRIMARY KEY (chat_id, message_id)
);
CREATE INDEX IF NOT EXISTS delivered_parent ON delivered (parent_id);
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
        # WAL: readers don't block the writer (and vice versa). Stored in the database file, so once is enough.
        c.execute("PRAGMA journal_mode=WAL")
        c.executescript(SCHEMA)
        cols = {r["name"] for r in c.execute("PRAGMA table_info(users)")}
        for col in ("backend", "model"):  # databases created before per-user models
            if col not in cols:
                c.execute(f"ALTER TABLE users ADD COLUMN {col} TEXT")
        if "daily_limit" not in cols:  # databases created before per-user daily limits
            c.execute("ALTER TABLE users ADD COLUMN daily_limit INTEGER")
        if "ocr_limit" not in cols:  # …and before the OCR limit
            c.execute("ALTER TABLE users ADD COLUMN ocr_limit INTEGER")
        if "tts_limit" not in cols:  # …and before the voice-message limit
            c.execute("ALTER TABLE users ADD COLUMN tts_limit INTEGER")
        if "ask_limit" not in cols:  # …and before the question limit
            c.execute("ALTER TABLE users ADD COLUMN ask_limit INTEGER")
        for col in ("unit_system", "temperature"):  # …and before per-user units (NULL = the default)
            if col not in cols:
                c.execute(f"ALTER TABLE users ADD COLUMN {col} TEXT")
        if "ocr" not in {r["name"] for r in c.execute("PRAGMA table_info(requests)")}:
            c.execute("ALTER TABLE requests ADD COLUMN ocr INTEGER DEFAULT 0")  # 1: this request ran OCR
        if "job" not in {r["name"] for r in c.execute("PRAGMA table_info(requests)")}:
            # The queued job as JSON (url, kind, options), so a stopped or cancelled one can be tried again,
            # also after a restart (the queue itself lives in memory).
            c.execute("ALTER TABLE requests ADD COLUMN job TEXT")
        if "source_url" not in {r["name"] for r in c.execute("PRAGMA table_info(uploads)")}:
            c.execute("ALTER TABLE uploads ADD COLUMN source_url TEXT")  # a Drive/Dropbox link instead of a file
        if "toc" not in {r["name"] for r in c.execute("PRAGMA table_info(documents)")}:
            c.execute("ALTER TABLE documents ADD COLUMN toc INTEGER DEFAULT 0")  # chapters from a contents list
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


def get_setting(key: str) -> str | None:
    """A value from the settings table (set by admins at runtime), or None if never set."""
    with _db() as c:
        row = c.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    return row["value"] if row else None


def set_setting(key: str, value: str) -> None:
    """Stores a setting (overwrites any earlier value)."""
    with _db() as c:
        c.execute("INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                  (key, value))


def add_upload(user_id: int, file_id: str, file_unique_id: str | None, name: str, size: int | None) -> int:
    """Records an uploaded file, before the user picks how to summarize it. Returns the upload id."""
    with _db() as c:
        return c.execute("INSERT INTO uploads (user_id, file_id, file_unique_id, name, size, created_at) "
                         "VALUES (?,?,?,?,?,?)", (user_id, file_id, file_unique_id, name, size, time.time())).lastrowid


def add_link_upload(user_id: int, url: str, name: str) -> int:
    """Records a document shared by link (Google Drive / Dropbox). Returns the upload id."""
    with _db() as c:
        return c.execute("INSERT INTO uploads (user_id, file_id, name, source_url, created_at) VALUES (?,?,?,?,?)",
                         (user_id, "", name, url, time.time())).lastrowid


def set_upload_name(upload_id: int, name: str) -> None:
    """Sets an upload's file name once it is known (a Drive link only reveals it when downloaded)."""
    with _db() as c:
        c.execute("UPDATE uploads SET name=? WHERE id=?", (name, upload_id))


def get_upload(upload_id: int) -> dict | None:
    """One upload row, or None."""
    with _db() as c:
        row = c.execute("SELECT * FROM uploads WHERE id=?", (upload_id,)).fetchone()
    return dict(row) if row else None


def set_upload_sha(upload_id: int, sha256: str) -> None:
    """Links an upload to its document once the file has been downloaded and hashed."""
    with _db() as c:
        c.execute("UPDATE uploads SET sha256=? WHERE id=?", (sha256, upload_id))


def get_document(sha256: str) -> dict | None:
    """A document's metadata (chapters decoded), or None. The text is in document_pages."""
    with _db() as c:
        row = c.execute("SELECT * FROM documents WHERE sha256=?", (sha256,)).fetchone()
    if not row:
        return None
    doc = dict(row)
    doc["chapters"] = json.loads(doc["chapters"]) if doc["chapters"] else []
    return doc


def save_document(sha256: str, **fields) -> None:
    """Creates or updates a document's metadata (chapters may be given as a list)."""
    if "chapters" in fields and not isinstance(fields["chapters"], (str, type(None))):
        fields["chapters"] = json.dumps(fields["chapters"], ensure_ascii=False)
    now = time.time()
    with _db() as c:
        c.execute("INSERT OR IGNORE INTO documents (sha256, status, created_at, updated_at) VALUES (?,?,?,?)",
                  (sha256, fields.get("status", "processing"), now, now))
        if fields:
            cols = ", ".join(f"{k}=?" for k in fields)
            c.execute(f"UPDATE documents SET {cols}, updated_at=? WHERE sha256=?", (*fields.values(), now, sha256))


def save_pages(sha256: str, pages: dict[int, str], source: str) -> None:
    """Stores page texts (replacing earlier ones for the same pages)."""
    with _db() as c:
        c.executemany("INSERT OR REPLACE INTO document_pages (sha256, page, text, source) VALUES (?,?,?,?)",
                      [(sha256, n, text, source) for n, text in pages.items()])


def get_pages(sha256: str) -> dict[int, str]:
    """All stored page texts of a document, by page index."""
    with _db() as c:
        return {r["page"]: r["text"] for r in c.execute(
            "SELECT page, text FROM document_pages WHERE sha256=? ORDER BY page", (sha256,))}


def get_doc_summaries(sha256: str, style: str, backend: str, model: str) -> dict[str, dict]:
    """Cached summaries of a document in one style by one model, by part ("book" or a chapter index)."""
    with _db() as c:
        return {r["part"]: json.loads(r["result"]) for r in c.execute(
            "SELECT part, result FROM doc_summaries WHERE sha256=? AND style=? AND backend=? AND model=?",
            (sha256, style, backend, model))}


def save_doc_summary(sha256: str, part: str, style: str, backend: str, model: str, result: dict) -> None:
    """Caches one summary of a document part (written as each is produced, so a re-run continues)."""
    with _db() as c:
        c.execute("INSERT OR REPLACE INTO doc_summaries VALUES (?,?,?,?,?,?,?)",
                  (sha256, part, style, backend, model, json.dumps(result, ensure_ascii=False), time.time()))


def get_request(req_id: int) -> dict | None:
    """One requests row, or None."""
    with _db() as c:
        row = c.execute("SELECT * FROM requests WHERE id=?", (req_id,)).fetchone()
    return dict(row) if row else None


def save_ocr_hold(request_id: int, upload_id: int, mode: str, chapter: int | None, pages: int, seconds: float,
                  language: str) -> None:
    """Remembers a document request that waits for the user's OCR confirmation (replacing an earlier one)."""
    with _db() as c:
        c.execute("INSERT OR REPLACE INTO ocr_holds (request_id, upload_id, mode, chapter, pages, seconds, language,"
                  " asked, approved, created_at) VALUES (?,?,?,?,?,?,?, COALESCE((SELECT asked FROM ocr_holds "
                  "WHERE request_id=?), 0), COALESCE((SELECT approved FROM ocr_holds WHERE request_id=?), 0), ?)",
                  (request_id, upload_id, mode, chapter, pages, seconds, language, request_id, request_id,
                   time.time()))


def get_ocr_hold(request_id: int) -> dict | None:
    """The OCR confirmation a request waits for, or None."""
    with _db() as c:
        row = c.execute("SELECT * FROM ocr_holds WHERE request_id=?", (request_id,)).fetchone()
    return dict(row) if row else None


def update_ocr_hold(request_id: int, **fields) -> None:
    """Updates an OCR hold (asked / approved)."""
    with _db() as c:
        c.execute(f"UPDATE ocr_holds SET {', '.join(f'{k}=?' for k in fields)} WHERE request_id=?",
                  (*fields.values(), request_id))


def get_page_sources(sha256: str) -> dict[int, str]:
    """Where each stored page's text came from ("text" layer or "ocr-<engine>"), by page index."""
    with _db() as c:
        return {r["page"]: r["source"] for r in c.execute(
            "SELECT page, source FROM document_pages WHERE sha256=?", (sha256,))}


# What counts toward each limit (access.LIMITS), as a filter on the user's requests in the window:
# - daily: every link, file or book request; voice messages and questions don't count (they have their own).
# - ocr: requests that started a full OCR run (finished or not; a cached OCR text costs nothing).
# - voice: newly made voice messages (reused ones are free); queued ones count only once they run.
# - ask: follow-up questions, queued ones included (so queueing several can't overrun the limit); cancelled
#   and failed ones count too, as for OCR.
USAGE_FILTERS = {
    "daily": "kind NOT IN ('voice','ask')",
    "ocr": "ocr=1",
    "voice": "kind='voice' AND cached=0 AND status<>'queued'",
    "ask": "kind='ask'",
}
_USER_LIMIT_COLUMNS = {"daily": "daily_limit", "ocr": "ocr_limit", "voice": "tts_limit", "ask": "ask_limit"}


def usage(uid: int, kind: str, window: float = 86400) -> tuple[int, float | None]:
    """How much of one limit a user used in the last `window` seconds, and when the oldest counted request was
    made (to tell when a slot frees up).

    Returns:
        (count, unix time of the oldest counted request or None).
    """
    with _db() as c:
        row = c.execute(f"SELECT COUNT(*) AS n, MIN(created_at) AS oldest FROM requests WHERE user_id=? "
                        f"AND created_at>? AND {USAGE_FILTERS[kind]}", (uid, time.time() - window)).fetchone()
    return row["n"], row["oldest"]


def set_user_limit(uid: int, kind: str, value: int | None) -> bool:
    """Sets (or with None removes) one user's override of a limit. Returns False if the user is unknown."""
    with _db() as c:
        return c.execute(f"UPDATE users SET {_USER_LIMIT_COLUMNS[kind]}=? WHERE id=?", (value, uid)).rowcount > 0


def get_user_units(uid: int) -> tuple[str, str]:
    """A user's units: (metric|imperial, c|f), the configured defaults where unset."""
    with _db() as c:
        row = c.execute("SELECT unit_system, temperature FROM users WHERE id=?", (uid,)).fetchone()
    system = (row["unit_system"] if row else None) or config.UNIT_SYSTEM
    temperature = (row["temperature"] if row else None) or config.TEMPERATURE
    return (system if system in ("metric", "imperial") else "metric", temperature if temperature in ("c", "f") else "c")


def set_user_units(uid: int, system: str | None = None, temperature: str | None = None) -> None:
    """Changes a user's measurement system and/or temperature scale (only the values given)."""
    with _db() as c:
        if system:
            c.execute("UPDATE users SET unit_system=? WHERE id=?", (system, uid))
        if temperature:
            c.execute("UPDATE users SET temperature=? WHERE id=?", (temperature, uid))


def save_spoken(request_id: int, title: str, text: str, lang: str, voice: str) -> None:
    """Stores what a summary's 🔊 button reads (replacing an earlier version for the same request)."""
    with _db() as c:
        c.execute("INSERT OR REPLACE INTO spoken VALUES (?,?,?,?,?,?)",
                  (request_id, title, text, lang, voice, time.time()))


def get_spoken(request_id: int) -> dict | None:
    """What a summary's 🔊 button reads, or None."""
    with _db() as c:
        row = c.execute("SELECT * FROM spoken WHERE request_id=?", (request_id,)).fetchone()
    return dict(row) if row else None


def get_voice(key: str, max_age: float) -> dict | None:
    """A made voice message by its key, unless older than `max_age` seconds (old ones are deleted here)."""
    with _db() as c:
        c.execute("DELETE FROM voices WHERE created_at<?", (time.time() - max_age,))
        row = c.execute("SELECT * FROM voices WHERE key=?", (key,)).fetchone()
    return dict(row) if row else None


def save_voice(key: str, file_id: str, duration: int | None) -> None:
    """Remembers Telegram's id of a sent voice message."""
    with _db() as c:
        c.execute("INSERT OR REPLACE INTO voices VALUES (?,?,?,?)", (key, file_id, duration, time.time()))


def delete_voice(key: str) -> None:
    """Forgets a voice message (Telegram no longer accepts its id)."""
    with _db() as c:
        c.execute("DELETE FROM voices WHERE key=?", (key,))


def waiting_request(user_id: int, sha256: str) -> int | None:
    """The user's latest request on this document that is waiting for a chapter to be picked, if any."""
    with _db() as c:
        row = c.execute("SELECT id FROM requests WHERE user_id=? AND platform='document' AND video_id=? "
                        "AND status='waiting' ORDER BY id DESC LIMIT 1", (user_id, sha256)).fetchone()
    return row["id"] if row else None


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
        # A document being read when the bot stopped: its saved pages stay, so a new request resumes.
        c.execute("UPDATE documents SET status='failed', error='bot restarted' WHERE status='processing'")
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


def save_delivered(request_id: int, kind: str, title: str, text: str, backend: str | None = None,
                   model: str | None = None, parent_id: int | None = None) -> None:
    """Records what a request showed the user (the 💬 and 📄 buttons work from it).

    Args:
        request_id: The request whose result was delivered.
        kind: video | file | document | ask.
        title: Its title (for an answer: the question).
        text: The text shown, before unit conversion.
        backend: The resolved backend that wrote it.
        model: The resolved model that wrote it.
        parent_id: For an answer, the summary request it was about.
    """
    with _db() as c:
        c.execute("INSERT OR REPLACE INTO delivered VALUES (?,?,?,?,?,?,?,?)",
                  (request_id, parent_id, kind, title, text, backend, model, time.time()))


def get_delivered(request_id: int) -> dict | None:
    """What a request showed the user, or None."""
    with _db() as c:
        row = c.execute("SELECT * FROM delivered WHERE request_id=?", (request_id,)).fetchone()
    return dict(row) if row else None


def ask_history(parent_id: int, user_id: int, n: int = 3) -> list[dict]:
    """The last n questions and answers about a summary by one user, oldest first.

    Only that user's: an admin asking about someone's summary mustn't change what the user's next answer sees.
    """
    with _db() as c:
        rows = c.execute("""SELECT d.title, d.text FROM delivered d JOIN requests r ON r.id=d.request_id
                            WHERE d.parent_id=? AND r.user_id=? ORDER BY d.request_id DESC LIMIT ?""",
                         (parent_id, user_id, n)).fetchall()
    return [dict(r) for r in reversed(rows)]


def save_messages(chat_id: int, message_ids: list[int], request_id: int) -> None:
    """Remembers bot messages a reply can point at, and which summary request a reply to them asks about."""
    with _db() as c:
        c.executemany("INSERT OR REPLACE INTO messages VALUES (?,?,?)",
                      [(chat_id, m, request_id) for m in message_ids])


def message_request(chat_id: int, message_id: int) -> int | None:
    """The summary request a bot message belongs to (see save_messages), or None."""
    with _db() as c:
        row = c.execute("SELECT request_id FROM messages WHERE chat_id=? AND message_id=?",
                        (chat_id, message_id)).fetchone()
    return row["request_id"] if row else None


def last_video_link(user_id: int) -> str | None:
    """The link of a user's latest YouTube/TikTok request (for /again or /transcript without a link), or None."""
    with _db() as c:
        row = c.execute("""SELECT url FROM requests WHERE user_id=? AND platform IN ('youtube', 'tiktok')
                           ORDER BY id DESC LIMIT 1""", (user_id,)).fetchone()
    return row["url"] if row else None
