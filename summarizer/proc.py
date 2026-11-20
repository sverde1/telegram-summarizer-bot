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
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
import time

from . import config

log = logging.getLogger(__name__)

# Each job has its own cancel event, set as this context variable while the job runs: asyncio.to_thread copies
# the context into the job's threads, and pool() into its thread pools. Every run() polls it, so a cancel stops
# that job's download/transcode/LLM call within ~0.5 s, and only that job's: jobs run side by side.
# Outside a job (the update check, /ocrlang) the default applies, an event nobody ever sets.
_NEVER = threading.Event()
job_cancel: ContextVar[threading.Event] = ContextVar("job_cancel", default=_NEVER)
# A pool's own stop (see pool): ends its other tasks' programs without marking the whole job cancelled.
_pool_stop: ContextVar[threading.Event] = ContextVar("pool_stop", default=_NEVER)


def cancel_event() -> threading.Event:
    """The current job's cancel event (one nobody sets, outside a job)."""
    return job_cancel.get()


def _stopped() -> bool:
    """Whether the current job was cancelled, or its pool stopped."""
    return job_cancel.get().is_set() or _pool_stop.get().is_set()


def pool(workers: int, stop: threading.Event | None = None) -> ThreadPoolExecutor:
    """A thread pool whose threads belong to the current job: its cancel (and an optional stop) reach them.

    Use it for every pool inside a job: plain ThreadPoolExecutor threads don't inherit the job's context, so
    a cancel would never reach their programs.

    Args:
        workers: Threads.
        stop: An event that stops the pool's programs too (e.g. after one task failed), without cancelling
            the job.
    """
    event, halt = cancel_event(), stop or _NEVER

    def init() -> None:
        """Puts the job's cancel event (and the pool's stop) into each new thread's context."""
        job_cancel.set(event)
        _pool_stop.set(halt)

    return ThreadPoolExecutor(max_workers=workers, initializer=init)

_POLL = 0.5  # seconds between checks for cancellation and the deadline


class ProcError(RuntimeError):
    """Base class for a program that was stopped by this module."""


class ProcTimeout(ProcError):
    """The program ran past its time limit and was killed."""


class ProcStalled(ProcTimeout):
    """The program printed nothing for its idle limit and was killed (a hung network call, e.g. Codex).

    Attributes:
        stdout: What it printed before (text), e.g. to find the session it started.
    """

    def __init__(self, message: str, stdout: str = ""):
        """Keeps the output printed before the stall."""
        super().__init__(message)
        self.stdout = stdout


class ProcCancelled(ProcError):
    """The job was cancelled while the program ran, so it was killed. Never swallow this: it must reach the
    worker, which stops the job."""


def check_cancelled() -> None:
    """Raises ProcCancelled if the current job was cancelled (a checkpoint between steps that run no program).

    Raises:
        ProcCancelled: The job was cancelled.
    """
    if _stopped():
        raise ProcCancelled("cancelled")


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
        env: dict | None = None, cwd=None, idle_timeout: float | None = None) -> subprocess.CompletedProcess:
    """Runs a program to completion and captures its output.

    Args:
        cmd: Program and arguments (never a shell string).
        timeout: Seconds before the program's whole process group is killed.
        input: Data for stdin, if any.
        text: Decode output as text (False: bytes, e.g. raw audio samples).
        env: Environment for the program; None = config.clean_env() (everything except the bot's secrets).
        cwd: Working directory.
        idle_timeout: Seconds without any output (stdout or stderr) before the program counts as hung and
            is killed; None = no such limit.

    Returns:
        The finished process (returncode, stdout, stderr); a non-zero exit is not an error here.

    Raises:
        ProcStalled: No output for idle_timeout seconds (a ProcTimeout too).
        ProcTimeout: The time limit passed.
        ProcCancelled: The current job was cancelled.
        OSError: The program couldn't be started (e.g. not installed).
    """
    if idle_timeout:
        return _run_watched(cmd, timeout, input, text, env, cwd, idle_timeout)
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=text,
                         env=config.clean_env() if env is None else env, cwd=cwd,
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
        if _stopped():
            _kill(p)
            raise ProcCancelled("cancelled")
        if time.monotonic() > deadline:
            _kill(p)
            raise ProcTimeout(f"{os.path.basename(cmd[0])} took longer than {timeout:.0f} s")


def _run_watched(cmd: list[str], timeout: float, input: str | bytes | None, text: bool, env: dict | None, cwd,
                 idle_timeout: float) -> subprocess.CompletedProcess:
    """run() that also notices when the program stops printing; see run() for the arguments.

    communicate() can't tell when output last arrived, so threads move the bytes: one reader per pipe (noting
    the time of each chunk) and one writer for stdin (the input can be bigger than the pipe buffer). They only
    move bytes, so they don't need the job's cancel: the polling loop here checks it and kills the group,
    which ends their pipes.
    """
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                         env=config.clean_env() if env is None else env, cwd=cwd, start_new_session=True)
    out: list[bytes] = []
    err: list[bytes] = []
    last = [time.monotonic()]

    def pump(stream, chunks: list[bytes]) -> None:
        """Collects one pipe until it closes, noting when output arrives."""
        for data in iter(lambda: stream.read1(65536), b""):
            chunks.append(data)
            last[0] = time.monotonic()

    def feed() -> None:
        """Writes the input and closes stdin, so the program sees its end."""
        try:
            p.stdin.write(input.encode() if isinstance(input, str) else input)
            p.stdin.close()
        except OSError:
            pass  # the program exited (or was killed) without reading it all

    threads = [threading.Thread(target=pump, args=(p.stdout, out), daemon=True),
               threading.Thread(target=pump, args=(p.stderr, err), daemon=True)]
    if input is not None:
        threads.append(threading.Thread(target=feed, daemon=True))
    for t in threads:
        t.start()

    def decode(chunks: list[bytes]) -> str | bytes:
        """The collected output, as text when asked for."""
        data = b"".join(chunks)
        return data.decode(errors="replace") if text else data

    def stop() -> None:
        """Kills the whole process group and waits for the pipes to close."""
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            log.warning("process %s didn't exit after SIGKILL", p.pid)
        for t in threads:
            t.join(timeout=5)

    deadline = time.monotonic() + timeout
    while True:
        try:
            p.wait(timeout=_POLL)
            break
        except subprocess.TimeoutExpired:
            pass
        if _stopped():
            stop()
            raise ProcCancelled("cancelled")
        now = time.monotonic()
        if now > deadline:
            stop()
            raise ProcTimeout(f"{os.path.basename(cmd[0])} took longer than {timeout:.0f} s")
        if now - last[0] > idle_timeout:
            stop()
            raise ProcStalled(f"{os.path.basename(cmd[0])} printed nothing for {idle_timeout:.0f} s",
                              stdout=b"".join(out).decode(errors="replace"))
    for t in threads:
        t.join(timeout=5)  # the pipes close with the program (unless a child it left behind still holds them)
    return subprocess.CompletedProcess(cmd, p.returncode, decode(out), decode(err))
