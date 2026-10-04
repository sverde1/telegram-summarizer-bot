"""Test setup: isolation from the real bot, network/process guards, and shared fixtures.

The bot's modules act at import time (config reads .env and creates data dirs, db creates its schema, access
syncs the admins), and test collection imports them. So the environment is set up here at module level,
before anything from the bot is imported.
"""
import asyncio
import atexit
import collections
import itertools
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

import dotenv
import pytest

# Never read the real .env: protecting only the secrets would still leak every other setting (model, limits,
# Whisper options) into the tests and make them machine-dependent. config.py binds load_dotenv at its own
# import, so patching it here, first, is enough.
dotenv.load_dotenv = lambda *args, **kwargs: False

# One temp dir per xdist worker (each worker is its own process and imports this file itself).
_TMP = Path(tempfile.mkdtemp(prefix=f"tsb-{os.environ.get('PYTEST_XDIST_WORKER', 'main')}-"))
atexit.register(shutil.rmtree, _TMP, True)
os.environ.pop("ALLOWED_USER_IDS", None)
os.environ.update({
    "PTB_TIMEDELTA": "1",  # as bot.py sets it, but before PTB is imported by anything else
    "DATA_DIR": str(_TMP / "data"),
    "CODEX_HOME": str(_TMP / "codex-home"),
    "ADMIN_USER_IDS": "1",
    "LLM_BACKEND": "codex",
    "CODEX_MODEL": "gpt-test",
    # Tests that run real OCR use Tesseract: it starts in a fraction of RapidOCR's time. RapidOCR and the GPU
    # path have their own tests.
    "OCR_ENGINE": "tesseract",
    "OCR_DEVICE": "cpu",
    "TELEGRAM_BOT_TOKEN": "",
    "ANTHROPIC_API_KEY": "",
    "OPENAI_API_KEY": "",
})

from summarizer import config  # noqa: E402  (must come after the environment above)

# Abort the whole run rather than risk touching the real bot's data or using its token.
if config.DATA_DIR.resolve() == (config.ROOT / "data").resolve() or config.TELEGRAM_BOT_TOKEN:
    raise SystemExit("tests aren't isolated from the real bot; refusing to run")

import access  # noqa: E402
from helpers import SUMMARY, FakeConversation, meta  # noqa: E402
import bot  # noqa: E402
from tgbot import state  # noqa: E402
from summarizer import db, media, stats, summarize  # noqa: E402
from telegram import File as telegram_File  # noqa: E402  (after PTB_TIMEDELTA is set)

ADMIN_ID = 1
# Programs that work offline. bwrap is only used by summarizer.sandbox, which always unshares the network.
LOCAL_TOOLS = {"bwrap", "prlimit", "pdftotext", "pdfinfo", "pdftoppm", "tesseract"}


# ---------- isolation ----------

@pytest.fixture(autouse=True)
def fresh_state(tmp_path, monkeypatch):
    """Gives every test its own empty database and stats file, and resets in-memory bot state.

    A fresh database (rather than emptying tables) keeps the admin row that `sync_admins` creates, so
    tests about the admin see the same state as the real bot. The job queue is replaced because an
    asyncio.Queue binds to the event loop it's first awaited on, and every test gets a new loop.
    """
    monkeypatch.setattr(db, "PATH", tmp_path / "bot.sqlite3")
    db.init()
    db.sync_admins(access.ADMINS)
    monkeypatch.setattr(stats, "_FILE", tmp_path / "stats.json")
    state.reset()
    import threading
    from summarizer import proc
    monkeypatch.setattr(proc, "current_job_cancel", threading.Event())
    # OCR models live in the worker's data dir, shared by its tests: start each test with none added.
    for folder in ("tessdata", "rapidocr"):
        shutil.rmtree(config.DATA_DIR / folder, ignore_errors=True)
    from summarizer import ocr, tts
    monkeypatch.setattr(ocr, "_cuda_ok", None)
    monkeypatch.setattr(tts, "_ready", False)  # voice messages off unless a test turns them on
    monkeypatch.setattr(tts, "_cuda_ok", None)
    # Pretend the installed Codex knows every feature we disable (asking it would spawn codex).
    monkeypatch.setattr(summarize, "_codex_known_features", set(summarize.CODEX_DISABLED_FEATURES))


@pytest.fixture(autouse=True)
def no_network(request, monkeypatch):
    """Blocks the internet and external programs, unless the test is marked `network`.

    Sockets and DNS are blocked so no test can reach YouTube, Telegram or an LLM by accident. Child
    processes are limited to local tools (ffmpeg, bwrap with its network unshared, poppler, Tesseract) and
    this Python interpreter, because yt-dlp, gallery-dl, codex, claude or npm would do their own
    networking, which a socket patch can't see.
    """
    if request.node.get_closest_marker("network"):
        return

    def blocked(*args, **kwargs):
        """Refuses any network access."""
        raise RuntimeError("network access is blocked in tests (mark the test `network` to allow it)")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)

    real_init = subprocess.Popen.__init__

    def guarded_init(self, args, *a, **kw):
        """Starts only local tools (ffmpeg, the document sandbox, poppler, Tesseract) or this interpreter."""
        first = args[0] if isinstance(args, (list, tuple)) else str(args).split()[0]
        name = os.path.basename(str(first))
        if (name in ("ffmpeg", "ffprobe") or name.startswith("ffmpeg-") or str(first) == sys.executable
                or name in LOCAL_TOOLS):
            return real_init(self, args, *a, **kw)
        raise RuntimeError(f"running {name!r} is blocked in tests (mark the test `network` to allow it)")

    monkeypatch.setattr(subprocess.Popen, "__init__", guarded_init)


# ---------- Telegram ----------

class TelegramRecorder:
    """Stands in for the Telegram servers: records every Bot API call and returns canned answers.

    Patched in at `ExtBot._do_post`, the single point every API call passes through, so PTB itself (filters,
    handlers, error classes) runs for real.

    Attributes:
        calls: (method, data) for every call, in order.
        fail: method name -> exception to raise once (e.g. {"sendMessage": Forbidden("blocked")}).
        files: file_id -> local file that "Telegram" serves for it (getFile + download).
    """

    def __init__(self):
        """Starts with no calls and no planned failures."""
        self.calls: list[tuple[str, dict]] = []
        self.fail: dict[str, Exception] = {}
        self.files: dict[str, Path] = {}
        self._ids = itertools.count(1000)

    def sent(self, method: str) -> list[dict]:
        """Returns the data of every call to one API method, e.g. "sendMessage"."""
        return [data for m, data in self.calls if m == method]

    def texts(self) -> list[str]:
        """Returns every text the bot sent or edited, in order."""
        return [data.get("text", "") for m, data in self.calls if m in ("sendMessage", "editMessageText")]

    async def post(self, endpoint: str, data: dict, **kwargs):
        """Handles one Bot API call like Telegram would (minus the network).

        Raises:
            Exception: The failure planned for this method in `fail`, once.
        """
        self.calls.append((endpoint, data))
        if endpoint in self.fail:
            raise self.fail.pop(endpoint)
        if endpoint == "getMe":
            return {"id": 999, "is_bot": True, "first_name": "Test bot", "username": "test_bot"}
        if endpoint == "getFile":
            fid = data["file_id"]
            return {"file_id": fid, "file_unique_id": f"u-{fid}", "file_size": self.files[fid].stat().st_size,
                    "file_path": f"documents/{fid}"}
        if endpoint in ("sendMessage", "editMessageText", "sendDocument"):
            chat_id = data.get("chat_id", 0)
            return {"message_id": data.get("message_id") or next(self._ids), "date": 0,
                    "chat": {"id": chat_id, "type": "private"}, "text": data.get("text", "")}
        if endpoint == "sendVoice":
            n = next(self._ids)
            return {"message_id": n, "date": 0, "chat": {"id": data.get("chat_id", 0), "type": "private"},
                    "voice": {"file_id": f"VOICE{n}", "file_unique_id": f"u{n}", "duration": 42}}
        return True


@pytest.fixture
def telegram(monkeypatch) -> TelegramRecorder:
    """The fake Telegram servers for this test (see TelegramRecorder)."""
    rec = TelegramRecorder()

    async def do_post(self, endpoint, data, **kwargs):
        """Routes the bot's API call to the recorder."""
        return await rec.post(endpoint, data, **kwargs)

    async def download_to_drive(self, custom_path=None, **kwargs):
        """Serves the file registered for this file_id instead of downloading it."""
        target = Path(custom_path)
        shutil.copy(rec.files[self.file_id], target)
        return target

    monkeypatch.setattr(telegram_File, "download_to_drive", download_to_drive)

    from telegram.ext import ExtBot
    monkeypatch.setattr(ExtBot, "_do_post", do_post)
    return rec


_update_ids = itertools.count(1)


def _user(uid: int, name: str = "Test", username: str | None = None) -> dict:
    """A Telegram user object."""
    return {"id": uid, "is_bot": False, "first_name": name, **({"username": username} if username else {})}


def msg_update(uid: int, text: str, *, chat_type: str = "private", edited: bool = False,
               name: str = "Test", username: str | None = None, reply_to: int | None = None,
               reply_from: int = 999) -> dict:
    """Builds the JSON of an incoming message update (a /command gets its bot_command entity).

    Args:
        uid: Sender's user id; also the chat id for private chats.
        text: Message text.
        chat_type: "private", "group", "supergroup" or "channel".
        edited: Send it as an edited message instead of a new one.
        name: Sender's first name.
        username: Sender's @username, if any.
        reply_to: The id of the message this one replies to, if any.
        reply_from: Who wrote that message: the bot (999, as getMe says) by default.
    """
    chat = {"id": uid if chat_type == "private" else -100, "type": chat_type}
    message = {"message_id": next(_update_ids), "date": 0, "text": text, "from": _user(uid, name, username),
               "chat": chat}
    if reply_to is not None:
        author = ({"id": 999, "is_bot": True, "first_name": "Test bot"} if reply_from == 999
                  else _user(reply_from))
        message["reply_to_message"] = {"message_id": reply_to, "date": 0, "chat": chat, "from": author,
                                       "text": "earlier"}
    if text.startswith("/"):
        message["entities"] = [{"type": "bot_command", "offset": 0, "length": len(text.split()[0])}]
    return {"update_id": next(_update_ids), "edited_message" if edited else "message": message}


def doc_update(uid: int, name: str, size: int, *, file_id: str = "F1", caption: str | None = None) -> dict:
    """Builds the JSON of an incoming message with an uploaded file (Telegram "document")."""
    message = {"message_id": next(_update_ids), "date": 0, "from": _user(uid),
               "chat": {"id": uid, "type": "private"},
               "document": {"file_id": file_id, "file_unique_id": f"u-{file_id}", "file_name": name,
                            "file_size": size}}
    if caption:
        message["caption"] = caption
    return {"update_id": next(_update_ids), "message": message}


def media_update(uid: int, kind: str, *, file_id: str = "M1", size: int = 50_000, duration: int = 3,
                 caption: str | None = None, **fields) -> dict:
    """Builds the JSON of an incoming voice message, audio, video or video note (`kind`: voice, audio, video,
    video_note); extra fields (file_name, mime_type, ...) go into the attachment."""
    attachment = {"file_id": file_id, "file_unique_id": f"u-{file_id}", "file_size": size, "duration": duration}
    if kind == "video":
        attachment |= {"width": 320, "height": 240}
    if kind == "video_note":
        attachment |= {"length": 240}
    message = {"message_id": next(_update_ids), "date": 0, "from": _user(uid), "chat": {"id": uid, "type": "private"},
               kind: attachment | fields}
    if caption:
        message["caption"] = caption
    return {"update_id": next(_update_ids), "message": message}


def callback_update(uid: int, data: str, *, chat_type: str = "private") -> dict:
    """Builds the JSON of an inline-button press (callback query) update."""
    return {"update_id": next(_update_ids), "callback_query": {
        "id": str(next(_update_ids)), "from": _user(uid), "chat_instance": "test", "data": data,
        "message": {"message_id": next(_update_ids), "date": 0, "text": "buttons",
                    "chat": {"id": uid if chat_type == "private" else -100, "type": chat_type}}}}


def my_chat_member_update(by_uid: int, chat_type: str = "group", status: str = "member") -> dict:
    """Builds the update Telegram sends when the bot is added to (or removed from) a chat."""
    member = {"user": {"id": 999, "is_bot": True, "first_name": "Test bot"}}
    return {"update_id": next(_update_ids), "my_chat_member": {
        "chat": {"id": -100, "type": chat_type, "title": "Family chat"}, "from": _user(by_uid, "Eve"), "date": 0,
        "old_chat_member": {**member, "status": "left"}, "new_chat_member": {**member, "status": status}}}


@pytest.fixture
async def app(telegram):
    """A real python-telegram-bot Application with the bot's handlers, talking to the fake Telegram.

    No updater (no polling) and no post_init (no worker or update checker): tests drive it with
    `await app.process_update(...)` and run the worker themselves when they need it.
    """
    from telegram.ext import Application
    application = Application.builder().token("123:TEST").updater(None).build()
    bot.add_handlers(application)
    await application.initialize()
    yield application
    await application.shutdown()


async def send(app, update_json: dict) -> None:
    """Feeds one update (built by msg_update/callback_update) through the application's handlers."""
    from telegram import Update
    await app.process_update(Update.de_json(update_json, app.bot))


@pytest.fixture
def codex_home():
    """A logged-in bot Codex home with a model list like Codex keeps it."""
    config.CODEX_HOME.mkdir(parents=True, exist_ok=True)
    (config.CODEX_HOME / "auth.json").write_text("{}")
    models = [{"slug": "gpt-a", "display_name": "GPT-A", "description": "best", "visibility": "list",
               "priority": 2, "input_modalities": ["text", "image"]},
              {"slug": "gpt-hidden", "visibility": "hide", "priority": 1},
              {"slug": "gpt-b", "display_name": "GPT-B", "visibility": "list", "priority": 3,
               "input_modalities": ["text"]}]
    (config.CODEX_HOME / "models_cache.json").write_text(json.dumps({"models": models}))
    yield config.CODEX_HOME
    for f in ("auth.json", "models_cache.json"):
        (config.CODEX_HOME / f).unlink(missing_ok=True)


@pytest.fixture
def fake_media(monkeypatch):
    """Replaces all downloads: a captioned 10-minute YouTube video, no thumbnail."""
    calls = []
    monkeypatch.setattr(media, "probe", lambda v, *a, **k: calls.append("probe") or meta())
    monkeypatch.setattr(media, "fetch_captions",
                        lambda v, m, w: calls.append("captions") or ([(0.0, "word " * 400)], "en"))
    monkeypatch.setattr(media, "download_thumbnail", lambda m, w: None)
    return calls


@pytest.fixture
def llm(monkeypatch):
    """Installs a FakeConversation factory; returns the list of conversations created."""
    convs = []

    def factory(backend=None, model=None):
        """Creates a conversation answering turn 1 with a plain summary."""
        c = FakeConversation([{**SUMMARY, "needs_frames": False, "frame_moments": []}], model=model or "x")
        convs.append(c)
        return c

    monkeypatch.setattr(summarize, "conversation", factory)
    return convs


# ---------- media ----------

def _make_video(path: Path, video_source: str) -> Path:
    """Renders a 4-second test video with a sine-tone soundtrack using the bot's ffmpeg."""
    subprocess.run([config.FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i", video_source,
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=4", "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)], check=True, timeout=60)
    return path


@pytest.fixture(scope="session")
def tiny_video(tmp_path_factory) -> Path:
    """A 4 s, 320x240 test-pattern video (moving content) with audio."""
    return _make_video(tmp_path_factory.mktemp("media") / "tiny.mp4", "testsrc=duration=4:size=320x240:rate=10")


@pytest.fixture(scope="session")
def blue_video(tmp_path_factory) -> Path:
    """A 4 s solid-blue video with audio: every frame is identical (for de-duplication tests)."""
    return _make_video(tmp_path_factory.mktemp("media") / "blue.mp4", "color=c=blue:duration=4:size=320x240:rate=10")
