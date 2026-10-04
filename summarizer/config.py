"""All settings come from environment variables (loaded from .env)."""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Everything the bot creates (database, downloads, logins, temp files) is private to its user. Set first,
# before any directory or file below is created: this module runs at import, ahead of everything else.
os.umask(0o077)

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    """Reads a setting from the environment, stripped of surrounding whitespace.

    Args:
        name: Variable name.
        default: Value when the variable isn't set.

    Returns:
        The value; an empty string if unset and no default given.
    """
    return os.environ.get(name, default).strip()


# Telegram
TELEGRAM_BOT_TOKEN = _env("TELEGRAM_BOT_TOKEN")

# LLM
LLM_BACKEND = _env("LLM_BACKEND", "codex")  # codex | claude-code | api
SUMMARY_LANGUAGE = _env("SUMMARY_LANGUAGE", "English")
# codex: ChatGPT subscription. Empty model = Codex's default.
CODEX_MODEL = _env("CODEX_MODEL")
CODEX_EFFORT = _env("CODEX_EFFORT", "medium")
# claude-code: Claude subscription. api: Anthropic API key.
CLAUDE_CODE_MODEL = _env("CLAUDE_CODE_MODEL", "claude-opus-5-5")
CLAUDE_MODEL = _env("CLAUDE_MODEL", "claude-opus-5-5")
CLAUDE_EFFORT = _env("CLAUDE_EFFORT", "medium")
# openai-api: OpenAI API key (pay per token, unlike the ChatGPT subscription Codex uses).
OPENAI_MODEL = _env("OPENAI_MODEL", "gpt-6-sol")
OPENAI_EFFORT = _env("OPENAI_EFFORT", "medium")

# Whisper (speech-to-text). CPU by default; set WHISPER_DEVICE=cuda once a GPU is installed.
WHISPER_DEVICE = _env("WHISPER_DEVICE", "cpu")  # cpu | cuda | auto
# An English-only *.en model would turn non-English speech into invented English instead of failing.
WHISPER_MODEL = _env("WHISPER_MODEL", "small")  # multilingual models only: small | medium | large-v3
# int8 is the fast CPU type; GPUs run float16.
WHISPER_COMPUTE_TYPE = _env("WHISPER_COMPUTE_TYPE", "int8" if WHISPER_DEVICE == "cpu" else "float16")
WHISPER_CPU_THREADS = int(_env("WHISPER_CPU_THREADS", "0"))  # 0 = ctranslate2 default

# Limits
MAX_DURATION_MIN = int(_env("MAX_DURATION_MIN", "180"))
# Per-user and global queue limits, and how often one user may redo (/again) the same video. Admins are exempt.
# They keep one account (careless or borrowed) from using up the LLM limits or blocking everyone for hours.
MAX_QUEUED_PER_USER = int(_env("MAX_QUEUED_PER_USER", "3"))  # queued + running
MAX_QUEUE = int(_env("MAX_QUEUE", "20"))
# Jobs worked on at the same time. Lookups, downloads and AI calls wait on the network and run fine side by
# side; the heavy CPU steps (Whisper, OCR, frame sweeps, voice) take turns anyway (summarizer/cpu.py).
WORKERS = max(1, int(_env("WORKERS", "8")))
# YouTube lookups (probe, captions) at the same time: a burst of them gets the server's IP flagged as a bot
# ("Sign in to confirm you're not a bot"). Downloads from a looked-up video aren't counted.
YOUTUBE_PARALLEL = max(1, int(_env("YOUTUBE_PARALLEL", "2")))
AGAIN_COOLDOWN_MIN = int(_env("AGAIN_COOLDOWN_MIN", "10"))
# Links a non-admin may send per rolling 24 hours (0 = no limit). Only the starting value: once an admin
# changes it with /limit, the value stored in the database wins.
DAILY_LIMIT = int(_env("DAILY_LIMIT", "100"))
# Uploaded documents: limits on what is read (a 2000-page PDF or 3 million characters is already huge; the
# caps keep a single upload from tying up the bot for hours).
MAX_DOC_PAGES = int(_env("MAX_DOC_PAGES", "2000"))
MAX_DOC_CHARS = int(_env("MAX_DOC_CHARS", "3000000"))
# Characters of book text sent to the AI in one call (~75k tokens; fits every backend's context with room for
# the answer). Longer books and chapters are summarized in pieces, then combined.
BOOK_CHUNK_CHARS = int(_env("BOOK_CHUNK_CHARS", "300000"))
# AI calls for one book run side by side (chapter batches): the machine mostly waits for the AI meanwhile.
BOOK_PARALLEL = int(_env("BOOK_PARALLEL", "3"))
# Text recognition for scanned documents: auto (RapidOCR on a GPU, Tesseract on the CPU, where it measured 5x
# faster), rapidocr or tesseract. Tesseract, when installed, also checks a scan's script.
OCR_ENGINE = _env("OCR_ENGINE", "auto")
# cpu or cuda (NVIDIA GPU, RapidOCR only; needs onnxruntime-gpu, falls back to the CPU when CUDA isn't there).
OCR_DEVICE = _env("OCR_DEVICE", "cpu")
# Scanned documents a non-admin may have read per rolling 24 h (0 = no limit); /limit ocr overrides it.
OCR_DAILY_LIMIT = int(_env("OCR_DAILY_LIMIT", "5"))
# Longest scan read without an admin's approval (pages needing OCR).
OCR_MAX_PAGES = int(_env("OCR_MAX_PAGES", "400"))
# OCR processes run in parallel; leave a core for the bot and Telegram.
OCR_WORKERS = int(_env("OCR_WORKERS", "3"))
# Largest document downloaded from a Google Drive / Dropbox link (Telegram itself stops bots at 20 MB).
MAX_LINK_DOWNLOAD_MB = int(_env("MAX_LINK_DOWNLOAD_MB", "100"))
# Largest audio/video file downloaded from such a link (only the audio is decoded; the file is deleted after).
MAX_MEDIA_LINK_MB = int(_env("MAX_MEDIA_LINK_MB", "1024"))
# The system's espeak-ng (Kokoro's text-to-speech needs it to turn text into phonemes):
# sudo apt install espeak-ng-data. Paths as on Debian/Ubuntu x86-64.
ESPEAK_LIB = _env("ESPEAK_LIB", "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1")
ESPEAK_DATA = _env("ESPEAK_DATA", "/usr/lib/x86_64-linux-gnu/espeak-ng-data")
# Voice messages (🔊 Listen). TTS_VOICE "" = the default voice for SUMMARY_LANGUAGE (see tts.LANGUAGES).
TTS_VOICE = _env("TTS_VOICE", "")
TTS_SPEED = float(_env("TTS_SPEED", "1.0"))  # 0.5-2.0
TTS_DEVICE = _env("TTS_DEVICE", "cpu")  # cpu, or cuda (needs onnxruntime-gpu; falls back to the CPU)
VOICE_CACHE_DAYS = int(_env("VOICE_CACHE_DAYS", "7"))  # how long a made voice message is reused
# New voice messages a non-admin may have made per rolling 24 h (0 = no limit); reused ones are free.
TTS_DAILY_LIMIT = int(_env("TTS_DAILY_LIMIT", "20"))
# Follow-up questions (💬 Ask) per user per 24 h; each resends the transcript, so they aren't free.
ASK_DAILY_LIMIT = int(_env("ASK_DAILY_LIMIT", "20"))
# What a question may send to the AI in all (summary, earlier answers, transcript or book text): the size books
# already use per call on every backend. Longer material is cut, a long book falls back to its summaries.
ASK_MAX_CHARS = int(_env("ASK_MAX_CHARS", str(BOOK_CHUNK_CHARS)))
ASK_MAX_QUESTION = int(_env("ASK_MAX_QUESTION", "1000"))  # characters; a question isn't an essay
# Default units for measurements in summaries (each user can change theirs with /units): metric or imperial,
# and c or f for temperatures.
UNIT_SYSTEM = _env("UNIT_SYSTEM", "metric")
TEMPERATURE = _env("TEMPERATURE", "c")
MAX_FRAMES = int(_env("MAX_FRAMES", "16"))
SHORT_VIDEO_SEC = int(_env("SHORT_VIDEO_SEC", "180"))  # up to this long: denser frame sampling
# Whisper may use at most this share of the machine's RAM; a video that doesn't fit right now waits (other
# jobs go first) for up to WHISPER_RAM_WAIT_MIN minutes. See summarizer/memory.py.
WHISPER_RAM_FRACTION = float(_env("WHISPER_RAM_FRACTION", "0.5"))
WHISPER_RAM_WAIT_MIN = int(_env("WHISPER_RAM_WAIT_MIN", "60"))
MAX_SLIDES = int(_env("MAX_SLIDES", "35"))  # TikTok carousels allow up to 35 images

DATA_DIR = Path(_env("DATA_DIR", str(ROOT / "data")))
DATA_DIR.mkdir(parents=True, exist_ok=True)
# The bot's own Codex login, separate from ~/.codex so the sandbox never sees your sessions/history.
# 0700: it holds the ChatGPT login tokens.
CODEX_HOME = Path(_env("CODEX_HOME", str(DATA_DIR / "codex-home")))
CODEX_HOME.mkdir(parents=True, exist_ok=True, mode=0o700)

# Resolve CLI tools next to the interpreter so a stale system yt-dlp never shadows the venv one,
# and so yt-dlp finds the venv's deno (YouTube JS runtime). Running .venv/bin/python (as the service
# does) doesn't put .venv/bin on PATH by itself.
VENV_BIN = Path(sys.executable).parent
os.environ["PATH"] = f"{VENV_BIN}{os.pathsep}{os.environ.get('PATH', '')}"

# The bot's own secrets. External programs never need them, and must not get them: Claude Code would bill
# ANTHROPIC_API_KEY instead of the Claude subscription, and a bug in a downloader parsing a hostile page
# shouldn't hand over the bot token.
_SECRET_VARS = {"TELEGRAM_BOT_TOKEN", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY"}


def clean_env() -> dict[str, str]:
    """The environment for external programs: this process's, minus the bot's secrets and any *_API_KEY.

    Built on each call (not once at import) so it reflects the current PATH and environment.
    """
    return {k: v for k, v in os.environ.items() if k not in _SECRET_VARS and not k.endswith("_API_KEY")}


def _tool(name: str) -> str:
    """Returns the venv's copy of a command-line tool, or the bare name to look up on PATH."""
    return str(VENV_BIN / name) if (VENV_BIN / name).exists() else name


YTDLP = _tool("yt-dlp")
GALLERY_DL = _tool("gallery-dl")


def _ffmpeg() -> str:
    """Finds an ffmpeg binary: the system one if installed, else the static build from imageio-ffmpeg.

    The fallback means the bot needs no system packages.

    Returns:
        Path of the ffmpeg executable.
    """
    import shutil
    if found := shutil.which("ffmpeg"):
        return found
    import imageio_ffmpeg  # bundled static build, used when ffmpeg isn't installed system-wide
    return imageio_ffmpeg.get_ffmpeg_exe()


FFMPEG = _ffmpeg()
# ffprobe comes with a system ffmpeg only (the imageio build has none); without it, ffmpeg -i is parsed.
FFPROBE = __import__("shutil").which("ffprobe")


def secure_files() -> None:
    """Makes the bot's secrets and data private to its user (other local accounts can't read them).

    umask only covers files created from now on; this fixes files that already exist with looser modes
    (e.g. a .env written by an editor as 664). Called once at startup, never at import, so tests only ever
    touch their own temp paths.
    """
    targets = [(ROOT / ".env", 0o600), (DATA_DIR, 0o700)]
    targets += [(p, 0o600) for p in DATA_DIR.glob("bot.sqlite3*")]  # the database and its -wal/-shm files
    for path, mode in targets:
        if path.exists() and (path.stat().st_mode & 0o777) != mode:
            path.chmod(mode)
