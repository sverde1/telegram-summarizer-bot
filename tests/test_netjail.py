"""The network jail: its commands, its fixed rules, and that it fails closed (the program never runs unjailed)."""
import os
import subprocess

import pytest

from summarizer import netjail


def test_bwrap_unshares_the_network_and_starts_blocked():
    cmd = netjail.bwrap_command(["--ro-bind", "/usr", "/usr"], ["node", "x.js"], 3, 4, 5)
    assert cmd[:5] == ["bwrap", "--unshare-all", "--unshare-user", "--disable-userns", "--die-with-parent"]
    assert cmd[cmd.index("--info-fd") + 1] == "3" and cmd[cmd.index("--block-fd") + 1] == "4"
    assert cmd[cmd.index("--file"):cmd.index("--file") + 3] == ["--file", "5", "/etc/resolv.conf"]
    assert cmd[-2:] == ["node", "x.js"] and "--share-net" not in cmd
    with pytest.raises(ValueError):
        netjail.bwrap_command(["--share-net"], ["node"], 3, 4, 5)


def test_slirp_and_nft_work_by_fd():
    slirp = netjail.slirp_command(7, 8, 9, 10)
    assert {"--disable-host-loopback", "--enable-seccomp", "--netns-type=path", "--configure"} <= set(slirp)
    assert "--enable-sandbox" not in slirp and "--userns-path=/proc/self/fd/7" in slirp
    assert slirp[-2:] == ["/proc/self/fd/8", "tap0"]
    nft = netjail.nft_command(7, 8)
    assert nft[:5] == ["nsenter", "--user=/proc/self/fd/7", "--net=/proc/self/fd/8", "--preserve-credentials",
                       "--keep-caps"] and nft[-3:] == ["/usr/sbin/nft", "-f", "-"]


def test_the_rules_allow_slirps_dns_and_reject_the_rest_locally():
    text = netjail.rules(["93.103.239.136"])
    assert 'oifname "lo" accept' in text and "ip daddr 10.0.2.3 udp dport 53 accept" in text
    for net in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "127.0.0.0/8", "169.254.0.0/16"):
        assert net in text
    assert "ip daddr { 93.103.239.136 } reject" in text and "meta nfproto ipv6 reject" in text
    assert "ip daddr {" in netjail.rules([]) and "93.103" not in netjail.rules([])


def test_run_args_runs_the_launcher_isolated():
    cmd = netjail.run_args(["--ro-bind", "/usr", "/usr"], ["codex", "exec"], python="/py")
    assert cmd[:3] == ["/py", "-I", netjail.__file__] and cmd[-3:] == ["--", "codex", "exec"]


class _Proc:
    """A fake child process."""

    def __init__(self, cmd, **kw):
        """Records how it was started; a fake slirp4netns says it's ready when told to."""
        self.cmd, self.killed, self.returncode = cmd, False, 0
        _Proc.started.append(self)
        if cmd[0] == "slirp4netns" and _Proc.slirp_ready:
            os.write(int(cmd[cmd.index("--ready-fd") + 1]), b"1")

    def kill(self):
        """Marks it killed."""
        self.killed = True

    def wait(self, timeout=None):
        """Already finished."""
        return self.returncode


@pytest.fixture
def fake(monkeypatch):
    """Fake bwrap/slirp4netns processes; the "sandbox" is this test's own process."""
    _Proc.started, _Proc.slirp_ready = [], True
    monkeypatch.setattr(netjail.subprocess, "Popen", _Proc)
    monkeypatch.setattr(netjail, "_read_pid", lambda fd: os.getpid())
    monkeypatch.setattr(netjail.fcntl, "ioctl", lambda fd, req: os.open("/proc/self/ns/user", os.O_RDONLY))
    monkeypatch.setattr(netjail, "_SETUP_TIMEOUT", 0.2)
    monkeypatch.setattr(netjail, "host_addresses", lambda: [])
    return _Proc


def test_it_fails_closed_when_slirp_isnt_ready(fake, capsys):
    fake.slirp_ready = False
    assert netjail.launch(["--ro-bind", "/usr", "/usr"], ["node"]) == netjail.JAIL_FAILED
    sandbox = fake.started[0]
    assert sandbox.cmd[0] == "bwrap" and sandbox.killed  # never unblocked: the program never ran
    assert "netjail: slirp4netns not ready" in capsys.readouterr().err


def test_it_fails_closed_when_the_rules_dont_load(fake, monkeypatch, capsys):
    monkeypatch.setattr(netjail.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, "", "Operation not permitted"))
    assert netjail.launch([], ["node"]) == netjail.JAIL_FAILED and fake.started[0].killed
    assert "nft failed: Operation not permitted" in capsys.readouterr().err


def test_with_the_jail_ready_the_program_runs(fake, monkeypatch):
    loaded = []
    monkeypatch.setattr(netjail.subprocess, "run", lambda cmd, **kw: loaded.append(kw["input"]) or
                        subprocess.CompletedProcess(cmd, 0, "", ""))
    fake.returncode = 0
    assert netjail.launch([], ["node"]) == 0 and not fake.started[0].killed
    assert "table inet jail" in loaded[0]


@pytest.mark.network
def test_the_real_jail_reaches_the_internet_only(monkeypatch):
    monkeypatch.undo()  # the conftest stub: run the real check
    assert netjail.self_check() == "" and netjail.ready()
