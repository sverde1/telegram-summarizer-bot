"""All settings come from environment variables (loaded from .env)."""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _env(name: str, default: str = "") -> str:
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

# Whisper (speech-to-text). CPU by default; set WHISPER_DEVICE=cuda once a GPU is installed.
WHISPER_DEVICE = _env("WHISPER_DEVICE", "cpu")  # cpu | cuda | auto
WHISPER_MODEL = _env("WHISPER_MODEL", "small")  # multilingual models only: small | medium | large-v3
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
CODEX_HOME = Path(_env("CODEX_HOME", str(DATA_DIR / "codex-home")))
CODEX_HOME.mkdir(parents=True, exist_ok=True, mode=0o700)

# Resolve CLI tools next to the interpreter so a stale system yt-dlp never shadows the venv one,
# and so yt-dlp finds the venv's deno (YouTube JS runtime).
VENV_BIN = Path(sys.executable).parent
os.environ["PATH"] = f"{VENV_BIN}{os.pathsep}{os.environ.get('PATH', '')}"


def _tool(name: str) -> str:
    return str(VENV_BIN / name) if (VENV_BIN / name).exists() else name


YTDLP = _tool("yt-dlp")
GALLERY_DL = _tool("gallery-dl")


def _ffmpeg() -> str:
    import shutil
    if found := shutil.which("ffmpeg"):
        return found
    import imageio_ffmpeg  # bundled static build, used when ffmpeg isn't installed system-wide
    return imageio_ffmpeg.get_ffmpeg_exe()


FFMPEG = _ffmpeg()
