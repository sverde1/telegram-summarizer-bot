"""The common result contract (summarizer.results.JobResult) for videos, recordings, documents and voice."""
from pathlib import Path

from summarizer import documents, pipeline, tts
from summarizer.results import JobResult

STATS = {"total": 42.0, "llm": "Codex · gpt-test"}


def _video(platform="youtube", url="https://youtu.be/x", meta=None, summary=None):
    """A video result with the given platform, URL/label, metadata and summary."""
    return pipeline.Result(platform, "x", url, meta or {"title": "Cats", "duration": 754}, "", "captions", "en",
                           summary)


def test_every_result_is_a_job_result():
    assert all(isinstance(r, JobResult) for r in (
        _video(), documents.DocResult("book", "b.pdf", {}), tts.VoiceResult("T", "text", "en-us", "af", "k")))


def test_heads_name_what_the_job_is_about():
    assert _video().head() == "🎬 Cats (12:34)"
    assert _video(meta={"title": "Pics", "duration": 30, "is_carousel": True}).head() == "🖼 Pics (photo post)"
    assert _video("file", "🎤 Voice message", {"title": "Audio file", "duration": 42}).head() == \
        "🎤 Voice message (0:42)"
    assert documents.DocResult("book", "b.pdf", {"pages": 12}).head() == "📄 b.pdf (12 pages)"
    assert pipeline.status_head("tiktok", "u", {"title": "P", "duration": 9}, photo=True) == "🖼 P (photo post)"


def test_only_fresh_summaries_report_work_time():
    assert _video(summary={"summary": "S", "_stats": STATS}).work_seconds() == 42.0
    assert _video(summary=None).work_seconds() is None  # a transcript-only run
    assert documents.DocResult("book", "b.pdf", {}).work_seconds() is None  # books don't teach the link estimate
    assert tts.VoiceResult("T", "text", "en-us", "af", "k").work_seconds() is None


def test_llm_label_comes_from_the_result():
    assert _video(summary={"summary": "S", "_stats": STATS}).llm_label() == "Codex · gpt-test"
    assert _video().llm_label() == ""
    assert documents.DocResult("book", "b.pdf", {}, llm="Claude · opus").llm_label() == "Claude · opus"


def test_shared_fields_are_keyword_only_and_default_off():
    r = tts.VoiceResult("T", "text", "en-us", "af", "k", ogg=Path("v.ogg"), cached=True)
    assert r.cached and r.hold is None and r.replay_steps is None and r.replay_total == 0.0
