"""The CPU slot: one heavy step at a time, first come first served, cancellable, waits not counted as work."""
import contextvars
import threading
import time

import pytest

from summarizer import cpu, proc


def _start(target):
    """Runs target in a thread with a fresh context (like another job) and returns the thread."""
    t = threading.Thread(target=lambda: contextvars.Context().run(target))
    t.start()
    return t


def test_one_holder_at_a_time_in_arrival_order():
    events, release = [], threading.Event()

    def holder():
        """Holds the slot until released."""
        with cpu.slot(None, "transcription", eta=60):
            events.append("holder in")
            release.wait(5)
        events.append("holder out")

    def waiter(name, delay):
        """Arrives after `delay` and takes its turn."""
        time.sleep(delay)
        with cpu.slot(None, "transcription"):
            events.append(name)

    threads = [_start(holder), _start(lambda: waiter("first", 0.1)), _start(lambda: waiter("second", 0.3))]
    time.sleep(0.6)
    assert events == ["holder in"]
    release.set()
    for t in threads:
        t.join(10)
    assert events == ["holder in", "holder out", "first", "second"]


def test_a_waiter_sees_why_and_roughly_how_long():
    shown, release, got_in = [], threading.Event(), threading.Event()

    def holder():
        """Holds the slot for a 100 s step."""
        with cpu.slot(None, "transcription", eta=100):
            got_in.set()
            release.wait(5)

    t = _start(holder)
    got_in.wait(2)
    threading.Timer(0.5, release.set).start()
    with cpu.slot(lambda text, eta: shown.append((text, eta)), "text recognition", eta=20):
        pass
    t.join(5)
    (text, eta), = shown
    assert text == "⏳ Waiting for a turn on the text recognition engine…" and 115 < eta <= 120


def test_a_cancel_stops_the_wait_and_the_next_one_still_gets_its_turn():
    release, got_in = threading.Event(), threading.Event()

    def holder():
        """Holds the slot."""
        with cpu.slot(None, "transcription"):
            got_in.set()
            release.wait(5)

    t = _start(holder)
    got_in.wait(2)
    threading.Timer(0.3, proc.cancel_event().set).start()
    with pytest.raises(proc.ProcCancelled):
        with cpu.slot(None, "transcription"):
            pass
    proc.cancel_event().clear()
    release.set()
    t.join(5)
    with cpu.slot(None, "transcription"):  # the cancelled ticket left the line
        pass


def test_released_after_an_error():
    with pytest.raises(RuntimeError):
        with cpu.slot(None, "voice"):
            raise RuntimeError("boom")
    with cpu.slot(None, "voice"):
        pass


def test_waits_are_left_out_of_the_jobs_timings():
    release, got_in = threading.Event(), threading.Event()

    def holder():
        """Holds the slot for 0.5 s."""
        with cpu.slot(None, "transcription"):
            got_in.set()
            release.wait(0.5)

    t = _start(holder)
    got_in.wait(2)
    cpu.track()
    since = time.monotonic()
    with cpu.slot(None, "transcription"):
        time.sleep(0.2)
    t.join(5)
    work = time.monotonic() - since - cpu.waited(since)
    assert 0.4 < cpu.waited() < 1 and 0.15 < work < 0.35
    assert cpu.waited(time.monotonic()) == 0  # nothing waited after now
