"""Is a newer Codex / Claude Code release out than the one installed? (The bot tells the admins.)"""
import logging
import re
import shutil

from . import proc

log = logging.getLogger(__name__)

# tool -> (command, npm package that publishes its releases, how to update)
# Both CLIs are published on npm, so `npm view <package> version` is one uniform way to learn the latest
# release, even for Claude Code installed by its native installer. Codex is installed globally as root,
# hence sudo; Claude Code lives in the user's home and updates itself.
TOOLS = {
    "Codex": ("codex", "@openai/codex", "sudo npm install -g @openai/codex"),
    "Claude Code": ("claude", "@anthropic-ai/claude-code", "claude update"),
}


def _version(text: str) -> tuple[int, ...] | None:
    """Extracts the first x.y.z version from command output.

    Args:
        text: E.g. "codex-cli 0.159.2" or "2.1.285 (Claude Code)".

    Returns:
        The version as a tuple of ints (so 0.10.0 compares above 0.9.0), or None if there is none.
    """
    m = re.search(r"(\d+)\.(\d+)\.(\d+)", text or "")
    return tuple(map(int, m.groups())) if m else None


def _run(cmd: list[str]) -> str:
    """Runs a command and returns its stdout.

    Returns:
        The output, or "" if the command is missing, fails to start or hangs: a failed check just
        means no notification this round.
    """
    try:
        return proc.run(cmd, timeout=60).stdout
    except (OSError, proc.ProcError) as e:
        log.warning("%s failed: %s", cmd[0], e)
        return ""


def check() -> list[dict]:
    """Finds installed CLIs that have a newer release.

    Returns:
        One dict per outdated tool: {tool, installed, latest, command}. Tools that aren't installed,
        or whose versions couldn't be read, are skipped.
    """
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
