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
MAX_FRAMES = int(_env("MAX_FRAMES", "16"))
SHORT_VIDEO_SEC = int(_env("SHORT_VIDEO_SEC", "180"))  # up to this long: denser frame sampling
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
