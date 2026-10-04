"""Parallel jobs on the same video or document: one works, the other waits and reuses its result."""
import threading
import time
from pathlib import Path

import pytest

from summarizer import config, media, pipeline, proc

URL = "https://youtu.be/abcdefghijk"


def test_a_second_job_waits_for_the_first():
    order, waited = [], []
    first_in = threading.Event()

    def first():
        """Holds the key for a moment."""
        with pipeline.one_at_a_time(("youtube", "x"), lambda: None):
            first_in.set()
            time.sleep(0.3)
            order.append("first")

    t = threading.Thread(target=first)
    t.start()
    first_in.wait(2)
    with pipeline.one_at_a_time(("youtube", "x"), lambda: waited.append(1)):
        order.append("second")
    t.join()
    assert order == ["first", "second"] and waited == [1]
    with pipeline.one_at_a_time(("youtube", "y"), lambda: waited.append(2)):  # other keys never wait
        pass
    assert waited == [1]


def test_a_cancel_stops_the_wait(monkeypatch):
    pipeline._busy.add(("youtube", "z"))
    try:
        proc.cancel_event().set()
        with pytest.raises(proc.ProcCancelled):
            with pipeline.one_at_a_time(("youtube", "z"), lambda: None):
                pass
    finally:
        pipeline._busy.discard(("youtube", "z"))
        proc.cancel_event().clear()


def test_the_same_video_twice_at_once_is_processed_once(fake_media, llm, monkeypatch):
    real_probe = media.probe

    def probe(*a, **k):
        """A slow lookup, so the second job arrives while the first works."""
        time.sleep(0.3)
        return real_probe(*a, **k)

    monkeypatch.setattr(media, "probe", probe)
    results = []
    jobs = [threading.Thread(target=lambda: results.append(pipeline.run(URL, lambda *a: None)))
            for _ in range(2)]
    for j in jobs:
        j.start()
        time.sleep(0.05)
    for j in jobs:
        j.join(10)
    assert sorted(r.cached for r in results) == [False, True] and len(llm) == 1


def test_every_job_gets_its_own_workdir(fake_media, llm, monkeypatch):
    seen = []
    real = pipeline.tempfile.mkdtemp

    def mkdtemp(**kw):
        """Records each job's directory."""
        seen.append(Path(real(**kw)))
        return str(seen[-1])

    monkeypatch.setattr(pipeline.tempfile, "mkdtemp", mkdtemp)
    pipeline.run(URL, lambda *a: None)
    pipeline.run(URL, lambda *a: None, use_cache=False)
    assert len(set(seen)) == 2 and all(p.parent == config.DATA_DIR / "work" for p in seen)
    assert not any(p.exists() for p in seen)  # each removed when its job finished
