"""Secrets and data are private to the bot's user."""
import os

from summarizer import config


def test_new_files_are_private(tmp_path):
    path = tmp_path / "new.txt"
    path.write_text("x")
    assert path.stat().st_mode & 0o777 == 0o600  # umask 077 set by config at import


def test_secure_files_fixes_existing_modes(tmp_path, monkeypatch):
    root, data = tmp_path / "root", tmp_path / "data"
    root.mkdir()
    data.mkdir()
    env, dbfile, wal = root / ".env", data / "bot.sqlite3", data / "bot.sqlite3-wal"
    for f in (env, dbfile, wal):
        f.write_text("x")
        f.chmod(0o664)
    data.chmod(0o775)
    monkeypatch.setattr(config, "ROOT", root)
    monkeypatch.setattr(config, "DATA_DIR", data)
    config.secure_files()
    assert [p.stat().st_mode & 0o777 for p in (env, dbfile, wal, data)] == [0o600, 0o600, 0o600, 0o700]


def test_secure_files_tolerates_missing_files(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ROOT", tmp_path / "nowhere")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "nodata")
    config.secure_files()  # nothing to fix, no error
    assert not os.path.exists(tmp_path / "nodata")
