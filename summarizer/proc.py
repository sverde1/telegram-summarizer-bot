"""Runs external programs (yt-dlp, gallery-dl, ffmpeg, Codex, Claude Code, npm) so they can always be stopped.

`subprocess.run(timeout=…)` kills only the direct child: yt-dlp's ffmpeg children survive, keep the output
pipes open (so the call can hang far past its timeout) and keep writing to disk. Here every program starts in
its own process group and the whole group is killed, on timeout or when the current job is cancelled.
Errors carry a short message, never the command line (it holds local paths and must not reach users).
"""
import logging
import os
import signal
import subprocess
import threading
import time

log = logging.getLogger(__name__)

# Set by the worker to cancel the job it's running (one worker runs one job at a time, so one event is
# enough). Every run() polls it, so a cancel stops the current download/transcode/LLM call within ~0.5 s.
current_job_cancel = threading.Event()

_POLL = 0.5  # seconds between checks for cancellation and the deadline


class ProcError(RuntimeError):
    """Base class for a program that was stopped by this module."""


class ProcTimeout(ProcError):
    """The program ran past its time limit and was killed."""


class ProcCancelled(ProcError):
    """The job was cancelled while the program ran, so it was killed. Never swallow this: it must reach the
    worker, which stops the job."""


def _kill(p: subprocess.Popen) -> None:
    """Kills the program's whole process group and reaps it.

    Codex runs inside bwrap with --unshare-pid, which makes bwrap the sandbox's init: killing bwrap's group
    takes everything inside the sandbox with it.
    """
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass  # already gone
    try:
        p.communicate(timeout=5)  # collect the exit status and drain the pipes
    except subprocess.TimeoutExpired:
        log.warning("process %s didn't exit after SIGKILL", p.pid)


def run(cmd: list[str], *, timeout: float, input: str | bytes | None = None, text: bool = True,
        env: dict | None = None, cwd=None) -> subprocess.CompletedProcess:
    """Runs a program to completion and captures its output.

    Args:
        cmd: Program and arguments (never a shell string).
        timeout: Seconds before the program's whole process group is killed.
        input: Data for stdin, if any.
        text: Decode output as text (False: bytes, e.g. raw audio samples).
        env: Environment for the program; None = this process's environment.
        cwd: Working directory.

    Returns:
        The finished process (returncode, stdout, stderr); a non-zero exit is not an error here.

    Raises:
        ProcTimeout: The time limit passed.
        ProcCancelled: The current job was cancelled.
        OSError: The program couldn't be started (e.g. not installed).
    """
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=text, env=env, cwd=cwd,
                         start_new_session=True)  # own process group, so _kill reaches its children too
    deadline = time.monotonic() + timeout
    pending_input = input
    while True:
        try:
            out, err = p.communicate(input=pending_input, timeout=_POLL)
            return subprocess.CompletedProcess(cmd, p.returncode, out, err)
        except subprocess.TimeoutExpired:
            # Input goes in on the first call only: communicate() refuses it once communication started.
            pending_input = None
        if current_job_cancel.is_set():
            _kill(p)
            raise ProcCancelled("cancelled")
        if time.monotonic() > deadline:
            _kill(p)
            raise ProcTimeout(f"{os.path.basename(cmd[0])} took longer than {timeout:.0f} s")
