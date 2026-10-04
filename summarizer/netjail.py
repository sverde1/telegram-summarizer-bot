"""The network jail: a bwrap sandbox with its own network that reaches the internet and nothing else.

Sandboxes that need the network (Codex, the PO-token script) used to share this machine's: besides the internet
they could reach its services (SSH, anything listening on its LAN address) and the LAN. Here each sandbox gets
its own network namespace, connected to the internet by slirp4netns, and an nftables rule set *inside* that
namespace rejects private, link-local and loopback destinations and this machine's own addresses. No root is
needed:

    python -I netjail.py <bwrap options> -- <command>     (or run_args() for Python callers)

The order is load-bearing. bwrap is started blocked (--block-fd); while it waits, slirp4netns attaches to its
network namespace and nft loads the rules, both through the user namespace that owns that network namespace
(NS_GET_USERNS on the netns fd; nsenter --keep-caps keeps the capabilities nft needs). Only then is the
program started. Once it runs, bwrap has moved it into a nested user namespace with no say over the network,
and it has no capabilities: it can't undo the rules. Anything failing before the start kills bwrap, so the
program never runs on an unfiltered network (exit code JAIL_FAILED, a "netjail:" line on stderr).

Standard library only, imports nothing from the bot: the PO-token wrapper runs it as a script with -I.
"""
import fcntl
import ipaddress
import json
import os
import select
import subprocess
import sys
import tempfile
import time

JAIL_FAILED = 197  # exit code when the jail couldn't be set up (the program didn't run)
NS_GET_USERNS = 0xB701  # ioctl: the user namespace owning a namespace (linux/nsfs.h)
DNS = "10.0.2.3"  # slirp4netns's DNS forwarder (to the host's resolver)
RESOLV_CONF = f"nameserver {DNS}\n"
_PRIVATE = ("0.0.0.0/8, 10.0.0.0/8, 100.64.0.0/10, 127.0.0.0/8, 169.254.0.0/16, 172.16.0.0/12, "
            "192.168.0.0/16, 224.0.0.0/4, 240.0.0.0/4")
_SETUP_TIMEOUT = 10  # seconds for each setup step


def rules(host_addresses: list[str]) -> str:
    """The fixed rule set, plus this machine's own addresses (taken from the host, never from a caller).

    Args:
        host_addresses: IPv4 addresses of this machine (e.g. a public one, or a VPN's), rejected too.
    """
    own = "".join(f"\n    ip daddr {{ {', '.join(host_addresses)} }} reject" if host_addresses else "")
    return f"""table inet jail {{
  chain out {{
    type filter hook output priority 0; policy accept;
    oifname "lo" accept
    ip daddr {DNS} udp dport 53 accept
    ip daddr {DNS} tcp dport 53 accept
    ip daddr {{ {_PRIVATE} }} reject{own}
    meta nfproto ipv6 reject
  }}
}}
"""


def host_addresses() -> list[str]:
    """This machine's IPv4 addresses that the private ranges don't already cover (e.g. a public address)."""
    try:
        out = subprocess.run(["ip", "-j", "-4", "addr"], capture_output=True, text=True, timeout=5).stdout
        found = {a["local"] for link in json.loads(out or "[]") for a in link.get("addr_info", [])}
    except (OSError, ValueError, subprocess.SubprocessError):
        return []
    nets = [ipaddress.ip_network(n.strip()) for n in _PRIVATE.split(",")]
    return sorted(a for a in found if not any(ipaddress.ip_address(a) in n for n in nets))


def bwrap_command(bwrap_args: list[str], command: list[str], info_fd: int, block_fd: int,
                  resolv_fd: int) -> list[str]:
    """The bwrap command: every namespace unshared (the network too), blocked until the jail is ready.

    Args:
        bwrap_args: The caller's mounts and settings (never a --share-net).
        command: What runs inside.
        info_fd: Where bwrap reports the sandbox's pid.
        block_fd: What bwrap waits on before starting the command.
        resolv_fd: The jail's resolv.conf contents (slirp's DNS).
    """
    if "--share-net" in bwrap_args:
        raise ValueError("--share-net would defeat the jail")
    return ["bwrap", "--unshare-all", "--unshare-user", "--disable-userns", "--die-with-parent",
            "--info-fd", str(info_fd), "--block-fd", str(block_fd), "--file", str(resolv_fd), "/etc/resolv.conf",
            *bwrap_args, *command]


def slirp_command(owner_fd: int, netns_fd: int, ready_fd: int, exit_fd: int) -> list[str]:
    """slirp4netns for the sandbox's network namespace, by fd (no pid that could be reused).

    --disable-host-loopback: the host's 127.0.0.1 isn't reachable through 10.0.2.2. --enable-seccomp: it parses
    packets from the sandbox. Not --enable-sandbox: it needs a uid 0, which a bwrap user namespace doesn't map.
    """
    return ["slirp4netns", "--configure", "--mtu=65520", "--disable-host-loopback", "--enable-seccomp",
            "--netns-type=path", f"--userns-path=/proc/self/fd/{owner_fd}", "--ready-fd", str(ready_fd),
            "--exit-fd", str(exit_fd), f"/proc/self/fd/{netns_fd}", "tap0"]


def nft_command(owner_fd: int, netns_fd: int) -> list[str]:
    """nft in the sandbox's network namespace, with the capabilities of the user namespace that owns it."""
    return ["nsenter", f"--user=/proc/self/fd/{owner_fd}", f"--net=/proc/self/fd/{netns_fd}",
            "--preserve-credentials", "--keep-caps", "/usr/sbin/nft", "-f", "-"]


def _read_pid(fd: int) -> int:
    """The sandbox's pid from bwrap's info fd (one JSON object).

    Raises:
        RuntimeError: bwrap didn't report one in time.
    """
    data, deadline = b"", time.monotonic() + _SETUP_TIMEOUT
    while b"}" not in data:
        if not select.select([fd], [], [], max(0.0, deadline - time.monotonic()))[0]:
            raise RuntimeError("bwrap didn't start in time")
        chunk = os.read(fd, 4096)
        if not chunk:
            raise RuntimeError("bwrap exited before starting the sandbox")
        data += chunk
    return int(json.loads(data[:data.index(b"}") + 1])["child-pid"])


def launch(bwrap_args: list[str], command: list[str]) -> int:
    """Runs a command in the jail and returns its exit code (JAIL_FAILED if the jail couldn't be set up).

    stdin, stdout and stderr are the caller's, untouched (Codex reads its prompt from stdin; the PO-token
    plugin parses node's stdout).
    """
    info_r, info_w = os.pipe()
    block_r, block_w = os.pipe()
    resolv_r, resolv_w = os.pipe()
    os.write(resolv_w, RESOLV_CONF.encode())
    os.close(resolv_w)
    sandbox = subprocess.Popen(bwrap_command(bwrap_args, command, info_w, block_r, resolv_r),
                               pass_fds=(info_w, block_r, resolv_r))
    for fd in (info_w, block_r, resolv_r):
        os.close(fd)
    slirp, exit_w, log = None, None, tempfile.TemporaryFile()
    try:
        pid = _read_pid(info_r)
        netns = os.open(f"/proc/{pid}/ns/net", os.O_RDONLY)
        owner = fcntl.ioctl(netns, NS_GET_USERNS)
        ready_r, ready_w = os.pipe()
        exit_r, exit_w = os.pipe()
        slirp = subprocess.Popen(slirp_command(owner, netns, ready_w, exit_r), stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=log, pass_fds=(owner, netns, ready_w, exit_r))
        os.close(ready_w)
        os.close(exit_r)
        if not select.select([ready_r], [], [], _SETUP_TIMEOUT)[0] or os.read(ready_r, 1) != b"1":
            log.seek(0)
            raise RuntimeError(f"slirp4netns not ready: {log.read().decode(errors='replace')[-300:]}")
        os.close(ready_r)
        nft = subprocess.run(nft_command(owner, netns), input=rules(host_addresses()), text=True,
                             capture_output=True, pass_fds=(owner, netns), timeout=_SETUP_TIMEOUT)
        if nft.returncode:
            raise RuntimeError(f"nft failed: {nft.stderr.strip()[-300:]}")
        os.close(netns)
        os.close(owner)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as e:
        sandbox.kill()  # the block fd was never written: the command never ran
        sandbox.wait()
        _stop_slirp(slirp, exit_w)
        print(f"netjail: {e}", file=sys.stderr)
        return JAIL_FAILED
    try:
        os.write(block_w, b"1")  # the jail is ready: start the command
    except BrokenPipeError:
        pass  # bwrap is already gone; its exit code says why
    os.close(block_w)
    code = sandbox.wait()
    _stop_slirp(slirp, exit_w)
    return code


def _stop_slirp(slirp: subprocess.Popen | None, exit_w: int | None) -> None:
    """Ends slirp4netns: it doesn't exit by itself when the sandbox is gone, only when its exit fd closes."""
    if exit_w is not None:
        os.close(exit_w)
    if slirp is not None:
        try:
            slirp.wait(5)
        except subprocess.TimeoutExpired:
            slirp.kill()
            slirp.wait()


def run_args(bwrap_args: list[str], command: list[str], python: str | None = None) -> list[str]:
    """The command line that runs `command` in the jail (for proc.run).

    Args:
        bwrap_args: bwrap's mounts and settings, without the bwrap binary or any --unshare/--share option.
        command: What runs inside.
        python: The interpreter to run this launcher with; this one by default.
    """
    return [python or sys.executable, "-I", os.path.abspath(__file__), *bwrap_args, "--", *command]


_PROBE = """
import socket, sys
def ok(host, port):
    try:
        socket.create_connection((host, port), 3).close()
        return True
    except OSError:
        return False
lan, gateway = sys.argv[1], sys.argv[2]
problems = []
if not ok("www.google.com", 443): problems.append("no internet from the jail")
if lan and ok(lan, 22): problems.append(f"this machine ({lan}:22) is reachable")
if gateway and ok(gateway, 53): problems.append(f"the LAN ({gateway}:53) is reachable")
print("; ".join(problems))
"""


def _route() -> tuple[str, str]:
    """(this machine's LAN address, the default gateway), "" where unknown."""
    try:
        out = subprocess.run(["ip", "-j", "-4", "route", "get", "1.1.1.1"], capture_output=True, text=True,
                             timeout=5).stdout
        r = json.loads(out or "[]")[0]
        return r.get("prefsrc", ""), r.get("gateway", "")
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return "", ""


_state: dict = {}  # {"ready": bool}, from self_check (the bot runs it once at startup)


def ready() -> bool:
    """Whether the last self_check passed (the jail works on this machine)."""
    return _state.get("ready", False)


def self_check() -> str:
    """Proves the jail works: from inside, the internet is reachable, this machine and the LAN aren't.

    Returns:
        "" when it works, else why not. Also remembered for ready().
    """
    why = _check()
    _state["ready"] = not why
    return why


def _check() -> str:
    """self_check's probe."""
    lan, gateway = _route()
    args = ["--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
            "--symlink", "usr/lib64", "/lib64", "--ro-bind-try", "/etc/nsswitch.conf", "/etc/nsswitch.conf",
            "--ro-bind-try", "/etc/hosts", "/etc/hosts", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
            "--clearenv", "--setenv", "PATH", "/usr/bin"]
    try:
        p = subprocess.run(run_args(args, ["/usr/bin/python3", "-I", "-c", _PROBE, lan, gateway]),
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        return f"the jail couldn't run: {e}"
    if p.returncode == JAIL_FAILED or p.returncode:
        return (p.stderr.strip().splitlines() or [f"exit code {p.returncode}"])[-1][:300]
    return p.stdout.strip()


def main(argv: list[str]) -> int:
    """Script entry: `netjail.py <bwrap options> -- <command>`."""
    if "--" not in argv:
        print("usage: netjail.py <bwrap options> -- <command>", file=sys.stderr)
        return 2
    i = argv.index("--")
    return launch(argv[:i], argv[i + 1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
