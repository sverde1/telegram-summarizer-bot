"""Book summaries: batching chapters, long chapters in parts, whole books, caching, short-mode budget."""
import pytest

from summarizer import books, config, db, summarize

from helpers import BookAI

SHA = "f" * 64


@pytest.fixture
def ai(monkeypatch):
    """Every summarize.conversation() is a fresh BookAI; returns the log of conversations."""
    BookAI.calls = []
    monkeypatch.setattr(summarize, "conversation", lambda backend=None, model=None: BookAI())
    return BookAI.calls


def _doc(chapter_sizes: list[int]) -> books.Books:
    """Stores a document with one page per chapter of the given sizes, and opens it for summarizing."""
    pages = {i: f"chapter {i} " + "x" * n for i, n in enumerate(chapter_sizes)}
    db.save_document(SHA, name="b.pdf", pages=len(pages), status="done",
                     chapters=[{"title": f"Ch <{i}>", "start": i, "end": i + 1} for i in pages])
    db.save_pages(SHA, pages, "text")
    seen = []
    b = books.Books(SHA, "codex", "gpt-test", lambda text, eta=None: seen.append(text))
    b.seen = seen
    return b


def test_chapters_are_batched_and_cached(ai, monkeypatch):
    monkeypatch.setattr(config, "BOOK_CHUNK_CHARS", 1000)
    b = _doc([300] * 6)
    assert b.chapters_summaries(list(range(6)), "short") == {i: f"S{i}" for i in range(6)}
    assert len(ai) == 2 and all(c.closed for c in ai)  # 3 chapters fit per call; one conversation each
    assert "&lt;0&gt;" in ai[0].text and "<0>" not in ai[0].text  # chapter titles from the file are escaped
    assert "summarizing 6 chapters" in b.seen[0] and any("chapters summarized: 3 of 6" in t for t in b.seen)
    assert b.chapters_summaries([2, 5], "short") == {2: "S2", 5: "S5"} and len(ai) == 2  # from the cache


def test_full_style_limits_chapters_per_call(ai):
    b = _doc([10] * 9)
    b.chapters_summaries(list(range(9)), "full")
    assert len(ai) == 3  # PER_CALL["full"] = 4: answers would get too long otherwise


def test_a_chapter_too_long_for_one_call_is_done_in_parts(ai, monkeypatch):
    monkeypatch.setattr(config, "BOOK_CHUNK_CHARS", 100)
    b = _doc([250])
    assert b.chapters_summaries([0], "full") == {0: "merged"}
    assert len(ai) == 4  # 3 parts + 1 merge


def test_ai_leaving_out_a_chapter_is_an_error(ai, monkeypatch):
    b = _doc([10, 10])

    class Lazy(BookAI):
        """Answers only the first chapter."""

        def _send(self, *a, **k):
            """Drops all chapters but the first."""
            answer = super()._send(*a, **k)
            return {"chapters": answer["chapters"][:1]}

    monkeypatch.setattr(summarize, "conversation", lambda backend=None, model=None: Lazy())
    with pytest.raises(summarize.SummaryError):
        b.chapters_summaries([0, 1], "short")
    assert db.get_doc_summaries(SHA, "short", "codex", "gpt-test") == {}  # nothing half-saved


def test_short_book_in_one_call(ai):
    b = _doc([100, 100])
    assert b.book() == {"title": "Book T", "author": "Ana", "summary": "Whole book."}
    assert len(ai) == 1 and "<document>" in ai[0].text
    b.book()
    assert len(ai) == 1  # cached


def test_long_book_from_its_short_chapter_summaries(ai, monkeypatch):
    monkeypatch.setattr(config, "BOOK_CHUNK_CHARS", 500)
    b = _doc([300] * 4)
    assert b.book()["summary"] == "Whole book."
    assert len(ai) == 5 and "S3" in ai[-1].text  # 4 chapter calls, then the book from their summaries
    assert len(db.get_doc_summaries(SHA, "short", "codex", "gpt-test")) == 4  # reusable for "short" mode


@pytest.mark.parametrize("n,budget", [(3, 900), (30, 289), (200, 250)])
def test_short_mode_budget_fits_three_messages(n, budget):
    assert books.short_budget(n) == budget


def test_ask_uses_a_fresh_closed_conversation_each_time(ai):
    summarize.ask("codex", None, "sys", "q", summarize.TEXT_SCHEMA)
    summarize.ask("codex", None, "sys", "q", summarize.TEXT_SCHEMA)
    assert len(ai) == 2 and all(c.closed for c in ai)


def test_a_failed_merge_keeps_the_parts(ai, monkeypatch):
    monkeypatch.setattr(config, "BOOK_CHUNK_CHARS", 100)
    b = _doc([250])  # 3 parts

    class FailingMerge(BookAI):
        """Summarizes parts, then fails the merge."""

        def _send(self, system, text, images, schema, first):
            """Raises on the merge call (the text-only schema)."""
            if "chapters" not in schema["properties"]:
                raise summarize.SummaryError(summarize.AI_FAILED, "merge failed")
            return super()._send(system, text, images, schema, first)

    monkeypatch.setattr(summarize, "conversation", lambda backend=None, model=None: FailingMerge())
    with pytest.raises(summarize.SummaryError):
        b.chapters_summaries([0], "full")
    assert len(ai) == 5  # 3 parts + the failed merge + its one retry (which reused the parts)
    monkeypatch.setattr(summarize, "conversation", lambda backend=None, model=None: BookAI())
    assert books.Books(SHA, "codex", "gpt-test", lambda *a: None).chapters_summaries([0], "short") == {0: "merged"}
    assert len(ai) == 6  # only the merge again: the parts came from the cache, for the other style too


def test_a_chapter_list_does_not_load_the_book(monkeypatch):
    _doc([10, 10])
    loaded = []
    monkeypatch.setattr(db, "get_pages", lambda sha: loaded.append(sha) or {})
    b = books.Books(SHA, "codex", "gpt-test", lambda *a: None)
    assert len(b.chapters) == 2 and loaded == []



# ---------- in parallel ----------

class SlowAI(BookAI):
    """A BookAI whose calls take a moment, recording how many run at the same time."""

    running = peak = 0
    guard = __import__("threading").Lock()

    def _send(self, *a, **k):
        """Answers like BookAI after a short wait, tracking concurrency."""
        import time
        with SlowAI.guard:
            SlowAI.running += 1
            SlowAI.peak = max(SlowAI.peak, SlowAI.running)
        deadline = time.monotonic() + 2  # wait until all three overlap (a loaded machine starts them late)
        while SlowAI.peak < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
        try:
            return super()._send(*a, **k)
        finally:
            with SlowAI.guard:
                SlowAI.running -= 1


def test_batches_run_side_by_side(ai, monkeypatch):
    SlowAI.running = SlowAI.peak = 0
    monkeypatch.setattr(summarize, "conversation", lambda backend=None, model=None: SlowAI())
    b = _doc([10] * 12)  # full style: 4 per call -> 3 calls
    assert b.chapters_summaries(list(range(12)), "full") == {i: f"S{i}" for i in range(12)}
    assert SlowAI.peak == 3 and len(db.get_doc_summaries(SHA, "full", "codex", "gpt-test")) == 12


def test_a_failed_batch_is_retried_once_and_the_others_kept(ai, monkeypatch):
    attempts = []

    class FlakyOnce(BookAI):
        """The batch with chapter 4 fails the first time."""

        def _send(self, system, text, images, schema, first):
            """Fails once for the second batch."""
            if 'index="4"' in text and not attempts:
                attempts.append(1)
                raise summarize.SummaryError(summarize.AI_FAILED, "flaky")
            return super()._send(system, text, images, schema, first)

    monkeypatch.setattr(summarize, "conversation", lambda backend=None, model=None: FlakyOnce())
    b = _doc([10] * 12)
    assert b.chapters_summaries(list(range(12)), "full") == {i: f"S{i}" for i in range(12)}
    assert len(ai) == 4  # 3 batches + 1 retry


def test_usage_limit_runs_the_rest_one_by_one(ai, monkeypatch):
    monkeypatch.setattr(config, "BOOK_PARALLEL", 1)  # deterministic order: the first call hits the limit
    calls = []

    class Limited(BookAI):
        """The first call hits the usage limit."""

        def _send(self, system, text, images, schema, first):
            """Raises AI_LIMIT once."""
            calls.append(1)
            if len(calls) == 1:
                raise summarize.SummaryError(summarize.AI_LIMIT, "limit")
            return super()._send(system, text, images, schema, first)

    monkeypatch.setattr(summarize, "conversation", lambda backend=None, model=None: Limited())
    b = _doc([10] * 8)
    assert b.chapters_summaries(list(range(8)), "full") == {i: f"S{i}" for i in range(8)}


def test_a_second_failure_fails_the_job(ai, monkeypatch):
    class Broken(BookAI):
        """Always fails the batch with chapter 0."""

        def _send(self, system, text, images, schema, first):
            """Fails for chapter 0, answers the rest."""
            if 'index="0"' in text:
                raise summarize.SummaryError(summarize.AI_FAILED, "broken")
            return super()._send(system, text, images, schema, first)

    monkeypatch.setattr(summarize, "conversation", lambda backend=None, model=None: Broken())
    b = _doc([10] * 8)
    with pytest.raises(summarize.SummaryError):
        b.chapters_summaries(list(range(8)), "full")
    assert len(db.get_doc_summaries(SHA, "full", "codex", "gpt-test")) == 4  # the other batch is kept


def test_cancel_stops_all_batches(ai, monkeypatch):
    from summarizer import proc

    class Cancelled(BookAI):
        """The job is cancelled during the calls."""

        def _send(self, *a, **k):
            """Raises like a killed call."""
            raise proc.ProcCancelled("cancelled")

    monkeypatch.setattr(summarize, "conversation", lambda backend=None, model=None: Cancelled())
    with pytest.raises(proc.ProcCancelled):
        _doc([10] * 12).chapters_summaries(list(range(12)), "full")
