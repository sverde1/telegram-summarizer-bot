"""CLI update checks."""
from summarizer import updates


def test_version_parsing():
    assert updates._version("codex-cli 0.159.2") == (0, 159, 2)
    assert updates._version("2.1.285 (Claude Code)") == (2, 1, 285)
    assert updates._version("nothing") is None


def test_check_reports_only_newer_releases(monkeypatch):
    monkeypatch.setattr(updates.shutil, "which", lambda name: f"/usr/bin/{name}")
    answers = {"codex": "codex-cli 0.159.2", "claude": "2.1.285 (Claude Code)",
               "@openai/codex": "0.160.0", "@anthropic-ai/claude-code": "2.1.285"}
    monkeypatch.setattr(updates, "_run", lambda cmd: answers[cmd[0] if cmd[0] != "npm" else cmd[2]])
    found = updates.check()
    assert [(u["tool"], u["installed"], u["latest"]) for u in found] == [("Codex", "0.159.2", "0.160.0")]
