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
    assert "chapters 1–3 of 6" in b.seen[0]
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
