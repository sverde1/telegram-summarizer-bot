"""PO tokens for YouTube: yt-dlp's arguments, the startup check, the sandboxed Node wrapper and installer."""
import logging
import subprocess
from pathlib import Path

import pytest

from summarizer import config, media, proc
from summarizer.urls import Video


YT = Video("youtube", "abcdefghijk", "https://www.youtube.com/watch?v=abcdefghijk")
TT = Video("tiktok", "123", "https://www.tiktok.com/@u/video/123")
ROOT = Path(__file__).resolve().parent.parent
WRAPPER = (ROOT / "deploy" / "node-sandboxed").read_text()
WRAPPER_CODE = "\n".join(line.split("  #")[0] for line in WRAPPER.splitlines() if not line.lstrip().startswith("#"))
INSTALLER = (ROOT / "deploy" / "install-pot.sh").read_text()


@pytest.fixture
def calls(monkeypatch):
    """Records yt-dlp commands and their extra environment instead of running them."""
    seen = []

    def run(cmd, timeout=900, env=None):
        """One fake yt-dlp run."""
        seen.append((cmd, env or {}))
        return subprocess.CompletedProcess(cmd, 0, '{"id": "x", "title": "T", "duration": 60}', "")

    monkeypatch.setattr(media, "_run", run)
    return seen


def test_youtube_calls_use_the_tokens_when_ready(calls, monkeypatch):
    monkeypatch.setattr(media, "_pot", {"ready": True})
    media.probe(YT)
    cmd, env = calls[-1]
    assert f"youtubepot-bgutilscript:server_home={config.POT_HOME}" in cmd
    assert cmd[cmd.index("--js-runtimes") + 1] == f"node:{ROOT / 'deploy' / 'node-sandboxed'}"
    assert "--no-warnings" not in cmd  # a failing token provider is only a warning: it must be seen
    assert env == {"POT_SERVER_HOME": str(config.POT_HOME), "POT_CACHE": str(config.POT_HOME.parent / "cache")}
    media.probe(TT)
    assert "--js-runtimes" not in calls[-1][0] and "--no-warnings" in calls[-1][0]  # TikTok: never


def test_without_the_setup_nothing_changes(calls):
    media.probe(YT)
    assert "--js-runtimes" not in calls[-1][0] and "--no-warnings" in calls[-1][0] and not calls[-1][1]


def test_warnings_reach_the_log_and_a_token_failure_explains_a_block(monkeypatch, caplog):
    err = ("WARNING: [youtube] [pot] Error fetching PO Token from \"bgutil:script-node\" provider\n"
           "ERROR: [youtube] x: Sign in to confirm you're not a bot")
    monkeypatch.setattr(proc, "run", lambda cmd, timeout, env=None: subprocess.CompletedProcess(cmd, 1, "", err))
    with caplog.at_level(logging.WARNING), pytest.raises(media.Blocked) as e:
        media._run(["yt-dlp"])
    assert "Error fetching PO Token" in caplog.text and "PO token: WARNING" in str(e.value)


@pytest.mark.parametrize("plugin, script, src, why", [
    (False, False, False, "not installed"),
    (True, False, False, "half installed"),
    (True, True, True, "must not exist"),
])
def test_check_pot_says_why_tokens_are_off(monkeypatch, tmp_path, plugin, script, src, why):
    import importlib.metadata as md
    monkeypatch.undo()  # the conftest stub, so the real check runs
    monkeypatch.setattr(config, "POT_HOME", tmp_path / "server")
    if script:
        (tmp_path / "server" / "build").mkdir(parents=True)
        (tmp_path / "server" / "build" / "generate_once.js").write_text("")
    if src:
        (tmp_path / "server" / "src").mkdir(parents=True)

    def version(name):
        """The plugin's version, or not installed."""
        if not plugin:
            raise md.PackageNotFoundError(name)
        return "2.0.1"

    monkeypatch.setattr(md, "version", version)
    assert why in media.check_pot() and not media.pot_ready()


def test_check_pot_runs_the_script_through_the_sandbox_and_compares_versions(monkeypatch, tmp_path):
    import importlib.metadata as md
    monkeypatch.undo()
    monkeypatch.setattr(config, "POT_HOME", tmp_path / "server")
    (tmp_path / "server" / "build").mkdir(parents=True)
    (tmp_path / "server" / "build" / "generate_once.js").write_text("")
    monkeypatch.setattr(md, "version", lambda name: "2.0.1")
    ran = []

    def run(cmd, timeout, env=None):
        """The wrapper answers with the script's version."""
        ran.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, answer[0], "")

    monkeypatch.setattr(proc, "run", run)
    answer = ["2.0.1\n"]
    assert media.check_pot() == "" and media.pot_ready()
    assert ran[-1][0] == str(ROOT / "deploy" / "node-sandboxed")
    answer[0] = "1.9.0\n"
    assert "1.9.0" in media.check_pot() and not media.pot_ready()


async def test_admins_hear_of_a_broken_setup_once_but_not_of_a_missing_one(app, telegram, monkeypatch):
    from tgbot import lifecycle
    monkeypatch.setattr(media, "check_pot", lambda: "not installed (deploy/install-pot.sh)")
    await lifecycle._check_pot(app)
    monkeypatch.setattr(media, "check_pot", lambda: "half installed: rerun deploy/install-pot.sh")
    await lifecycle._check_pot(app)
    await lifecycle._check_pot(app)  # the same problem after a restart: not again
    sent = [d["text"] for d in telegram.sent("sendMessage")]
    assert sent == ["⚠️ YouTube PO tokens are off: half installed: rerun deploy/install-pot.sh"]


def test_the_wrapper_sandboxes_node_with_network_only():
    code = WRAPPER_CODE
    assert "--unshare-all --share-net --die-with-parent" in code and "--clearenv" in code
    assert "--new-session" not in code  # stays in yt-dlp's process group, so a cancel kills it
    binds = [line.split()[1:3] for line in code.splitlines() if line.strip().startswith(("--bind ", "--ro-bind "))]
    assert ["/usr", "/usr"] in binds and ['"$server"', '"$server"'] in binds and ['"$cache"', "/cache"] in binds
    assert "/home" not in code and ".env" not in code and "ROOT" not in code
    assert 'exec flock "$cache/.lock" prlimit' in code and "/usr/bin/node --max-old-space-size" in code


def test_the_installer_is_pinned_sandboxed_and_keeps_no_sources():
    assert "COMMIT=" in INSTALLER and 'rev-parse HEAD' in INSTALLER
    assert "bwrap --unshare-all --share-net" in INSTALLER and "npm ci" in INSTALLER
    assert 'cp -a "$build/repo/server/build" "$build/repo/server/node_modules" "$build/repo/server/package.json"' \
        in INSTALLER
    assert 'test ! -e "$dest/src"' in INSTALLER  # the plugin would prefer src/'s deno, outside the sandbox


@pytest.mark.network
def test_a_real_token_is_minted_in_the_sandbox():
    """With deploy/install-pot.sh run: a YouTube lookup gets a PO token from the script through the wrapper.

    The web client is forced (it always wants a token); YouTube streams it in a format yt-dlp skips, hence
    --ignore-no-formats-error. A token from the script's 6 h cache counts too: it was minted the same way.
    """
    server = ROOT / "data" / "bgutil" / "server"
    if not (server / "build" / "generate_once.js").exists():
        pytest.skip("deploy/install-pot.sh hasn't been run")
    wrapper = ROOT / "deploy" / "node-sandboxed"
    env = config.clean_env() | {"POT_SERVER_HOME": str(server), "POT_CACHE": str(server.parent / "cache")}
    p = subprocess.run([config.YTDLP, "-v", "--simulate", "--no-playlist", "--ignore-no-formats-error",
                        "--extractor-args", f"youtubepot-bgutilscript:server_home={server}",
                        "--extractor-args", "youtube:player_client=web;fetch_pot=always",
                        "--js-runtimes", f"node:{wrapper}",
                        "https://www.youtube.com/watch?v=jNQXAC9IVRw"], capture_output=True, text=True, timeout=180,
                       env=env)
    assert f"Executing command to get POT via script: {wrapper} " in p.stderr
    assert "Retrieved a gvs PO Token for web client" in p.stderr
    assert "bgutil:script-deno-2.0.1 (external, unavailable)" in p.stderr
