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
    event = proc.cancel_event()
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


def _in_job(event: threading.Event, fn):
    """Runs fn in a fresh context whose job cancel event is `event` (like a job's thread)."""
    import contextvars

    def body():
        """Sets the job's event, then runs fn."""
        proc.job_cancel.set(event)
        return fn()

    return contextvars.Context().run(body)


def test_a_cancel_stops_only_its_own_jobs_program():
    a, b = threading.Event(), threading.Event()
    results = {}

    def job(name, event, seconds):
        """Runs a sleeping program as job `name`; records how it ended."""
        try:
            _in_job(event, lambda: proc.run([PY, "-c", f"import time; time.sleep({seconds})"], timeout=60))
            results[name] = "finished"
        except proc.ProcCancelled:
            results[name] = "cancelled"

    threads = [threading.Thread(target=job, args=("a", a, 30)), threading.Thread(target=job, args=("b", b, 1.5))]
    for t in threads:
        t.start()
    time.sleep(0.3)
    a.set()
    for t in threads:
        t.join(10)
    assert results == {"a": "cancelled", "b": "finished"}


def test_pool_threads_belong_to_the_job_and_a_pool_stop_doesnt_cancel_it():
    event, stop = threading.Event(), threading.Event()

    def in_job():
        """Starts a pool program, cancels the job, and reports how fast the program stopped."""
        with proc.pool(1) as pool:
            future = pool.submit(proc.run, [PY, "-c", "import time; time.sleep(30)"], timeout=60)
            time.sleep(0.3)
            start = time.monotonic()
            event.set()
            with pytest.raises(proc.ProcCancelled):
                future.result(10)
            return time.monotonic() - start

    assert _in_job(event, in_job) < 3

    def pool_stop():
        """A pool's own stop ends its programs but leaves the job running."""
        with proc.pool(1, stop) as pool:
            future = pool.submit(proc.run, [PY, "-c", "import time; time.sleep(30)"], timeout=60)
            time.sleep(0.3)
            stop.set()
            with pytest.raises(proc.ProcCancelled):
                future.result(10)
        return proc.cancel_event().is_set()

    assert _in_job(threading.Event(), pool_stop) is False


def test_outside_a_job_nothing_cancels_a_program():
    import contextvars
    proc.cancel_event().set()  # the test's own "job" is cancelled
    p = contextvars.Context().run(lambda: proc.run([PY, "-c", "print('ok')"], timeout=10))
    assert p.stdout.strip() == "ok" and not contextvars.Context().run(proc.cancel_event).is_set()
