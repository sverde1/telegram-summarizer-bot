"""The killable process runner."""
import sys
import threading
import time

import pytest

from summarizer import proc

PY = sys.executable


def test_output_input_and_exit_code():
    p = proc.run([PY, "-c", "import sys; print(sys.stdin.read().upper()); sys.exit(3)"], timeout=10, input="hi")
    assert (p.returncode, p.stdout.strip()) == (3, "HI")


def test_bytes_output():
    p = proc.run([PY, "-c", "import sys; sys.stdout.buffer.write(bytes([0, 255]))"], timeout=10, text=False)
    assert p.stdout == b"\x00\xff"


def _pid_alive(pid: int) -> bool:
    """Whether a process with this id still exists (and isn't a zombie)."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split()[2] != "Z"
    except FileNotFoundError:
        return False


def test_timeout_kills_children_too(tmp_path):
    pidfile = tmp_path / "child.pid"
    # The program starts a grandchild that would outlive a plain subprocess.run timeout and hold the pipes.
    script = (f"import subprocess, sys, time; c = subprocess.Popen([sys.executable, '-c', 'import time; "
              f"time.sleep(60)']); open({str(pidfile)!r}, 'w').write(str(c.pid)); time.sleep(60)")
    start = time.monotonic()
    with pytest.raises(proc.ProcTimeout) as e:
        proc.run([PY, "-c", script], timeout=1.5)
    assert time.monotonic() - start < 5
    assert PY not in str(e.value)  # no command line in the message
    time.sleep(0.2)
    assert not _pid_alive(int(pidfile.read_text()))


def test_cancel_stops_the_program_quickly(monkeypatch):
    event = threading.Event()
    monkeypatch.setattr(proc, "current_job_cancel", event)
    threading.Timer(0.3, event.set).start()
    start = time.monotonic()
    with pytest.raises(proc.ProcCancelled):
        proc.run([PY, "-c", "import time; time.sleep(30)"], timeout=60)
    assert time.monotonic() - start < 3



def test_programs_never_get_the_bots_secrets(monkeypatch):
    for name in ("TELEGRAM_BOT_TOKEN", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "SOME_API_KEY"):
        monkeypatch.setenv(name, "secret-value")
    monkeypatch.setenv("HARMLESS", "ok")
    p = proc.run([PY, "-c", "import os, json; print(json.dumps(dict(os.environ)))"], timeout=10)
    import json
    child = json.loads(p.stdout)
    assert "secret-value" not in p.stdout
    assert child["HARMLESS"] == "ok" and "PATH" in child
