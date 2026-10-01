"""The test setup itself: tests must never touch the real bot, the network or external programs."""
import subprocess
import urllib.request

import pytest

from summarizer import config, db, stats

from conftest import ADMIN_ID, msg_update, send


def test_data_dir_is_temporary():
    assert config.DATA_DIR.resolve() != (config.ROOT / "data").resolve()
    assert "tsb-" in str(config.DATA_DIR)


def test_real_env_not_loaded():
    assert config.TELEGRAM_BOT_TOKEN == ""
    assert config.CODEX_MODEL == "gpt-test"
    assert config.LLM_BACKEND == "codex"


def test_database_and_stats_are_per_test(tmp_path):
    assert db.PATH.parent == tmp_path
    assert stats._FILE.parent == tmp_path
    assert db.get_user(ADMIN_ID)["status"] == "admin"


def test_network_is_blocked():
    with pytest.raises(Exception, match="blocked in tests"):
        urllib.request.urlopen("http://127.0.0.1:9", timeout=2)


def test_external_programs_are_blocked():
    with pytest.raises(RuntimeError, match="blocked in tests"):
        subprocess.run(["yt-dlp", "--version"])


def test_ffmpeg_is_allowed(tiny_video):
    assert tiny_video.stat().st_size > 1000


async def test_fake_telegram_round_trip(app, telegram):
    await send(app, msg_update(ADMIN_ID, "/help"))
    assert any("YouTube or TikTok link" in t for t in telegram.texts())
