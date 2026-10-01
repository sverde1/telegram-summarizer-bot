"""Startup removes downloads and LLM session files left behind by a crashed run, and nothing else."""
from pathlib import Path

from summarizer import config, pipeline, summarize


def test_leftovers_are_removed(tmp_path, monkeypatch):
    data, home = tmp_path / "data", tmp_path / "home"
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "CODEX_HOME", data / "codex-home")
    monkeypatch.setattr(summarize.ClaudeCodeConversation, "CWD", data / "claude-cwd")
    monkeypatch.setattr(Path, "home", lambda: home)  # never the real ~/.claude
    project = home / ".claude" / "projects" / str(data / "claude-cwd").replace("/", "-")
    other_project = home / ".claude" / "projects" / "-some-other-project"
    leftovers = [data / "work" / "youtube_x" / "audio.m4a", data / "tmpabc123" / "in" / "img01.jpg",
                 data / "codex-home" / "sessions" / "2026" / "rollout-1.jsonl", project / "uuid-1.jsonl"]
    keep = [data / "bot.sqlite3", data / "stats.json", data / "codex-home" / "auth.json",
            other_project / "my-own-session.jsonl"]
    for f in leftovers + keep:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x")
    removed = pipeline.cleanup_leftovers()
    assert removed == 4
    assert not any(f.exists() for f in leftovers)
    assert all(f.exists() for f in keep)
    assert not (data / "work" / "youtube_x").exists() and not (data / "tmpabc123").exists()
