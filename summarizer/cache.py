"""SQLite cache keyed by (platform, video_id), so the same video is never fetched twice."""
import json
import sqlite3
import threading
import time

from . import config

_DB = config.DATA_DIR / "cache.sqlite3"
_lock = threading.Lock()


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(_DB)
    c.execute("""CREATE TABLE IF NOT EXISTS videos (
        platform TEXT, video_id TEXT, meta TEXT, transcript TEXT, transcript_source TEXT,
        language TEXT, result TEXT, frames_used INTEGER, created REAL,
        PRIMARY KEY (platform, video_id))""")
    return c


def get(platform: str, video_id: str) -> dict | None:
    with _lock, _conn() as c:
        row = c.execute("SELECT meta, transcript, transcript_source, language, result, frames_used "
                        "FROM videos WHERE platform=? AND video_id=?", (platform, video_id)).fetchone()
    if not row:
        return None
    return {"meta": json.loads(row[0]), "transcript": row[1], "transcript_source": row[2],
            "language": row[3], "result": json.loads(row[4]) if row[4] else None,
            "frames_used": bool(row[5])}


def put(platform: str, video_id: str, *, meta: dict, transcript: str, transcript_source: str,
        language: str, result: dict | None, frames_used: bool) -> None:
    with _lock, _conn() as c:
        c.execute("INSERT OR REPLACE INTO videos VALUES (?,?,?,?,?,?,?,?,?)",
                  (platform, video_id, json.dumps(meta), transcript, transcript_source, language,
                   json.dumps(result) if result else None, int(frames_used), time.time()))
