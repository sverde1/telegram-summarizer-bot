"""The CPU slot: heavy steps (Whisper, OCR, frame sweeps, Kokoro) take turns, everything else runs side by side.

Jobs run in parallel, but a heavy step already uses every core (and Whisper a lot of memory): two at once
would only slow each other down and blur the memory guard. A job reaching a heavy step while another holds the
slot waits for its turn, first come first served, showing that in its status; the wait can be cancelled.

Waiting is not work: every wait is recorded for the job (track / waited), and the timers that feed the
footer, the privacy pacing and the speed statistics leave it out.
"""
import threading
import time
from collections import deque
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar

from . import proc

_changed = threading.Condition()
_line: deque = deque()  # tickets of the jobs waiting for the slot, first come first served
_holder: dict = {}  # the current holder: {"until": monotonic time its step should end}; empty = free
# The waits of the current job, as (start, end) monotonic times (see track); None outside a tracked job.
_waits: ContextVar[list[tuple[float, float]] | None] = ContextVar("cpu_waits", default=None)


def track() -> None:
    """Starts recording this job's waits for the slot (call at the start of its work, in its thread)."""
    _waits.set([])


def waited(since: float = 0.0) -> float:
    """Seconds this job waited for the slot after `since` (a time.monotonic() value), to leave out of a timer."""
    return sum(end - max(start, since) for start, end in _waits.get() or [] if end > since)


def _wait_text(what: str) -> str:
    """The status line while waiting. Neutral: it mustn't tell users how busy others are."""
    return f"⏳ Waiting for a turn on the {what} engine…"


@contextmanager
def slot(status: Callable[[str, float | None], None] | None, what: str, eta: float | None = None):
    """Holds the CPU slot for one heavy step, waiting for it first if another job has it.

    Args:
        status: Callback (text, eta) for the job's status while it waits, or None.
        what: What the step is, for the waiting line ("transcription", "text recognition", ...).
        eta: Seconds this step should take, so jobs waiting behind it can be told roughly how long.

    Raises:
        proc.ProcCancelled: The job was cancelled while waiting.
    """
    ticket = object()
    start = time.monotonic()
    with _changed:
        _line.append(ticket)
        try:
            shown = False
            while _line[0] is not ticket or _holder:
                if not shown and status:
                    left = max(0.0, _holder["until"] - time.monotonic()) if _holder else None
                    status(_wait_text(what), left + (eta or 0) if left is not None else None)
                    shown = True
                _changed.wait(1)
                proc.check_cancelled()
        except BaseException:
            _line.remove(ticket)
            _changed.notify_all()
            raise
        _line.popleft()
        _holder["until"] = time.monotonic() + (eta or 0)
    end = time.monotonic()
    if end - start > 0.01 and (waits := _waits.get()) is not None:
        waits.append((start, end))
    try:
        yield
    finally:
        with _changed:
            _holder.clear()
            _changed.notify_all()
