"""The pipeline with fake media and a fake LLM: caching per model, frames requests, helpers."""
import pytest

from summarizer import db, media, pipeline, summarize

from helpers import SUMMARY, FakeConversation, meta

URL = "https://youtu.be/abcdefghijk"


def test_first_run_then_cache_hit(fake_media, llm):
    r1 = pipeline.run(URL, lambda *a: None)
    assert not r1.cached and r1.summary["summary"] == "A summary." and llm[0].closed
    assert r1.summary["_stats"]["model"] == "gpt-test"  # CODEX_MODEL from the test env
    r2 = pipeline.run(URL, lambda *a: None)
    assert r2.cached and len(llm) == 1 and fake_media.count("probe") == 1


def test_another_model_gets_its_own_summary_without_redownloading(fake_media, llm):
    pipeline.run(URL, lambda *a: None)
    r = pipeline.run(URL, lambda *a: None, backend="codex", model="other-model")
    assert not r.cached and len(llm) == 2
    assert fake_media.count("captions") == 1  # transcript reused from the database
    assert db.get_summary("youtube", "abcdefghijk", "codex", "other-model") is not None


def test_again_skips_the_cache(fake_media, llm):
    pipeline.run(URL, lambda *a: None)
    assert not pipeline.run(URL, lambda *a: None, use_cache=False).cached


def test_too_long_video_is_refused(monkeypatch, llm):
    monkeypatch.setattr(media, "probe", lambda v: meta(duration=10 ** 6))
    with pytest.raises(pipeline.PipelineError, match="longer than"):
        pipeline.run(URL, lambda *a: None)


def test_frames_requested_by_the_llm_go_into_turn_two(fake_media, monkeypatch, tmp_path):
    conv = FakeConversation([{**SUMMARY, "needs_frames": True, "frame_moments": [{"t": 30, "why": "book"}]},
                             {**SUMMARY, "summary": "With frames."}])
    monkeypatch.setattr(summarize, "conversation", lambda backend=None, model=None: conv)
    frame = tmp_path / "f.jpg"
    frame.write_bytes(b"jpg")
    monkeypatch.setattr(pipeline, "_frames", lambda *a, **k: [(frame, "t=0:31")])
    r = pipeline.run(URL, lambda *a: None)
    assert r.summary["summary"] == "With frames." and r.frames_used
    assert "needs_frames" not in conv.sent[1][1]  # turn 2 uses the plain summary schema


def test_progress_reports_stages(fake_media, llm):
    seen = []
    pipeline.run(URL, lambda text, eta=None: seen.append(text.splitlines()[-1]))
    assert seen[0].startswith("🔎") and any("Summarizing" in s for s in seen)


@pytest.mark.parametrize("words,duration,speechless", [(0, 60, True), (4, 600, True), (10, 60, True),
                                                       (100, 60, False), (20, 10, False)])
def test_speechless(words, duration, speechless):
    assert pipeline._speechless({"duration": duration}, "[0:00] " + "w " * words) is speechless


def test_duration_format_and_etas():
    assert pipeline._fmt_duration(75) == "1:15"
    assert pipeline._fmt_duration(3725) == "1:02:05"
    assert pipeline._eta_audio(200) == 4
    assert pipeline._eta_llm(0, 0) == 25  # starting guess before anything was measured


def test_reused_transcript_is_paced_for_first_time_requesters(fake_media, llm, monkeypatch):
    pipeline.run(URL, lambda *a: None)  # someone summarized it before
    paused = []
    monkeypatch.setattr(pipeline, "_pause", paused.append)
    seen = []
    r = pipeline.run(URL, lambda text, eta=None: seen.append(text), backend="codex", model="other-model",
                     hide_cache_from=60)
    assert not any("from cache" in s for s in seen) and any("Transcript ready" in s for s in seen)
    assert any("Checking for YouTube captions" in s for s in seen)  # the step a fresh run shows
    assert paused == [pipeline.CAPTIONS_SECONDS * pipeline.REPLAY_SHARE]
    assert [n for n, _ in r.replay_steps] == ["lookup", "captions", "summary"]  # footer: what the user saw
    saved = db.get_summary("youtube", "abcdefghijk", "codex", "other-model")["result"]["_stats"]
    assert [n for n, _ in saved["steps"]] == ["lookup", "summary"]  # stored: the real work only


def test_reused_transcript_is_not_paced_for_admins_or_repeat_requesters(fake_media, llm, monkeypatch):
    pipeline.run(URL, lambda *a: None)
    paused = []
    monkeypatch.setattr(pipeline, "_pause", paused.append)
    r = pipeline.run(URL, lambda *a: None, backend="codex", model="other-model")  # admin: hide_cache_from=None
    assert paused == [] and r.replay_steps is None


def test_pause_stops_at_once_on_cancel():
    from summarizer import proc
    proc.current_job_cancel.set()
    try:
        with pytest.raises(proc.ProcCancelled):
            pipeline._pause(60)
    finally:
        proc.current_job_cancel.clear()
