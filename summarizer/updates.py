"""Is a newer Codex / Claude Code release out than the one installed? (The bot tells the admins.)"""
import logging
import re
import shutil
import subprocess

log = logging.getLogger(__name__)

# tool -> (command, npm package that publishes its releases, how to update)
TOOLS = {
    "Codex": ("codex", "@openai/codex", "sudo npm install -g @openai/codex"),
    "Claude Code": ("claude", "@anthropic-ai/claude-code", "claude update"),
}


def _version(text: str) -> tuple[int, ...] | None:
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", text or "")
    return tuple(map(int, m.groups())) if m else None


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired) as e:
        log.warning("%s failed: %s", cmd[0], e)
        return ""


def check() -> list[dict]:
    """Tools with a newer release: [{tool, installed, latest, command}]."""
    out = []
    for tool, (cmd, package, how) in TOOLS.items():
        if not shutil.which(cmd):
            continue
        installed = _version(_run([cmd, "--version"]))
        latest = _version(_run(["npm", "view", package, "version"]))
        if installed and latest and latest > installed:
            out.append({"tool": tool, "installed": ".".join(map(str, installed)),
                        "latest": ".".join(map(str, latest)), "command": how})
    return out
