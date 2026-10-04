"""Summaries of uploaded books and documents: the whole book, every chapter (short or full), or one chapter.

Any book size fits any backend because the text goes to the AI in pieces of at most BOOK_CHUNK_CHARS:
- chapters are packed together into one call while they fit (far fewer calls on the owner's subscription);
- a chapter longer than a piece is summarized in parts, which are then merged;
- a whole book that doesn't fit one call is summarized from its (short) chapter summaries.
Every summary is cached by document, part, style and model as soon as it exists, so a later request (or a
re-run after a cancel or a restart) only does what is missing.
"""
import html
import math
import threading
import time
from collections.abc import Callable
from concurrent.futures import as_completed

from . import config, db, pipeline, proc, stats, summarize

TG_LIMIT = 4096
SHORT_MAX = 900            # characters per chapter in "short" mode, when there are only a few chapters
SHORT_MIN = 250            # …and the least it gets with many chapters
SHORT_MESSAGES = 3         # "short" mode should fit in this many messages
PER_CALL = {"short": 30, "full": 4}  # chapters per call: bounded by the answer's length, not the input's


def short_budget(n_chapters: int) -> int:
    """Characters per chapter in "short" mode, so all chapters fit in SHORT_MESSAGES messages."""
    overhead = 120  # chapter title line, spacing, the footer's share
    per = (SHORT_MESSAGES * TG_LIMIT) // max(n_chapters, 1) - overhead
    return max(SHORT_MIN, min(SHORT_MAX, per))


def chapter_text(pages: dict[int, str], chapter: dict) -> str:
    """The text of one chapter."""
    return "\n".join(pages.get(i, "") for i in range(chapter["start"], chapter["end"])).strip()


def _material(doc: dict) -> str:
    """The document's metadata, as context for the AI (untrusted, like its text)."""
    return (f"<document_info>File name: {doc.get('name') or ''}\nTitle in metadata: {doc.get('title') or ''}\n"
            f"Author in metadata: {doc.get('author') or ''}\nPages: {doc.get('pages')}</document_info>")


def _length(style: str, budget: int) -> str:
    """The length instruction for chapter summaries."""
    return summarize.CHAPTER_LENGTHS[style].format(budget=budget)


def _pieces(text: str, size: int) -> list[str]:
    """Cuts text into pieces of at most `size` characters, at paragraph or line breaks where possible."""
    pieces = []
    while len(text) > size:
        cut = max(text.rfind("\n\n", 0, size), text.rfind("\n", 0, size))
        cut = cut if cut > size // 2 else size
        pieces.append(text[:cut])
        text = text[cut:].lstrip()
    return pieces + [text] if text else pieces


class Books:
    """Summarizes one document with one user's backend and model, reporting progress as it goes.

    Args:
        sha256: The document.
        backend: The user's backend (resolved, not None).
        model: The user's model (resolved: the cache key).
        status: Callback (text, eta) for the status message; also a cancel checkpoint.
    """

    def __init__(self, sha256: str, backend: str, model: str, status: Callable[[str, float | None], None]):
        """Loads the document and its text."""
        self.sha, self.backend, self.model, self.status = sha256, backend, model, status
        self.doc = db.get_document(sha256)
        self._pages: dict[int, str] | None = None
        self.chapters = self.doc["chapters"]
        self.llm = summarize.llm_label(backend, model)
        self.steps: list[tuple[str, float]] = []  # (label, seconds) for the footer
        self._lock = threading.Lock()  # calls run in parallel threads (chapters_summaries)

    @property
    def pages(self) -> dict[int, str]:
        """The document's text by page, loaded on first use (a chapter list or a cached answer needs none)."""
        if self._pages is None:
            self._pages = db.get_pages(self.sha)
        return self._pages

    def _ask(self, system_extra: str, text: str, schema: dict) -> dict:
        """One AI call with timing, speed statistics and a cancel check afterwards."""
        t = time.monotonic()
        answer, answered_by = summarize.ask(self.backend, self.model, summarize.BOOK_SYSTEM,
                                            f"{system_extra}\n\n{text}", schema)
        proc.check_cancelled()  # an API call can't be interrupted; at least don't go on after it
        took = time.monotonic() - t
        stats.record(f"llm:{self.backend}", took / pipeline.llm_load(len(text), 0))
        with self._lock:
            if answered_by:
                self.llm = summarize.llm_label(self.backend, answered_by)
            self.steps.append(("summary", took))
        return answer

    def _eta(self, chars: int, calls: int = 1) -> float:
        """Estimated seconds for `calls` AI calls over `chars` characters in total."""
        return calls * pipeline.eta_llm(chars / max(calls, 1), 0, self.backend)

    def chapters_summaries(self, indices: list[int], style: str) -> dict[int, str]:
        """Summaries of the given chapters in one style ("short" or "full"), from the cache where possible.

        Returns:
            {chapter index: summary text}.

        Raises:
            summarize.SummaryError: The AI failed or left a chapter out.
        """
        budget = short_budget(len(self.chapters))
        done = {int(k): v["summary"] for k, v in db.get_doc_summaries(self.sha, style, self.backend, self.model).items()
                if k != "book"}
        todo = [i for i in indices if i not in done]
        chunk = config.BOOK_CHUNK_CHARS
        big = [i for i in todo if len(chapter_text(self.pages, self.chapters[i])) > chunk]
        batches, current, size = [], [], 0
        for i in todo:
            if i in big:
                continue
            n = len(chapter_text(self.pages, self.chapters[i]))
            if current and (size + n > chunk or len(current) >= PER_CALL[style]):
                batches.append(current)
                current, size = [], 0
            current.append(i)
            size += n
        if current:
            batches.append(current)
        units = [batch for batch in batches] + [[i] for i in big]  # a long chapter is a unit of its own
        if not units:
            return {i: done[i] for i in indices}
        total, lock = len(indices), threading.Lock()
        finished = [total - len(todo)]
        workers = min(len(units), max(1, config.BOOK_PARALLEL))
        per_call = self._eta(sum(len(chapter_text(self.pages, self.chapters[i])) for i in todo), len(units))

        def progress(left: int) -> None:
            """The status line: chapters done so far, and the time for the units still to do."""
            label = ("summarizing the chapter" if total == 1 else
                     f"chapters summarized: {finished[0]} of {total}" if finished[0] else f"summarizing {total} chapters")
            self.status(f"🧠 {self.llm}: {label}…", math.ceil(left / workers) * per_call)

        def run(unit: list[int]) -> None:
            """Summarizes one unit (a batch, or one long chapter) and caches its summaries at once."""
            if unit[0] in big:
                got = {unit[0]: self._long_chapter(unit[0], style, budget)}
            else:
                got = self._batch(unit, style, budget)
            for i, summary in got.items():
                db.save_doc_summary(self.sha, str(i), style, self.backend, self.model, {"summary": summary})
            with lock:
                done.update(got)
                finished[0] += len(got)

        progress(len(units))
        failed: list[list[int]] = []
        with proc.pool(workers) as pool:
            futures = {pool.submit(run, unit): unit for unit in units}
            left = len(units)
            for future in as_completed(futures):
                left -= 1
                if future.cancelled():
                    continue  # stopped after a usage limit; already queued to run one by one below
                error = future.exception()
                if isinstance(error, proc.ProcCancelled):
                    raise error  # the other calls see the same cancel and stop; the pool waits for them
                if isinstance(error, summarize.SummaryError):
                    # Calls already running are paid for: let them finish (their summaries are cached), and
                    # retry this unit once on its own afterwards. A usage limit makes the rest run one by one.
                    failed.append(futures[future])
                    if str(error) == summarize.AI_LIMIT:
                        for other, unit in futures.items():
                            if other.cancel():
                                failed.append(unit)
                elif error is not None:
                    raise error
                elif left:
                    progress(left)
        for unit in failed:  # once more, one at a time; a second failure fails the job
            progress(len(failed))
            run(unit)
        return {i: done[i] for i in indices}

    def _batch(self, batch: list[int], style: str, budget: int) -> dict[int, str]:
        """One AI call summarizing several chapters; every one of them must come back.

        Raises:
            summarize.SummaryError: The call failed or left a chapter out.
        """
        body = "\n".join(f'<chapter index="{i}" title="{html.escape(self.chapters[i]["title"])}">\n'
                         f'{chapter_text(self.pages, self.chapters[i])}\n</chapter>' for i in batch)
        answer = self._ask(summarize.CHAPTERS_PROMPT.format(length=_length(style, budget)),
                           f"{_material(self.doc)}\n{body}", summarize.CHAPTERS_SCHEMA)
        got = {e["index"]: e["summary"] for e in answer["chapters"] if e.get("index") in batch}
        if missing := [i for i in batch if not got.get(i, "").strip()]:
            raise summarize.SummaryError(summarize.AI_FAILED, f"the AI left out chapters {missing}")
        return {i: got[i] for i in batch}

    def _long_chapter(self, i: int, style: str, budget: int) -> str:
        """A chapter too long for one call: its parts are summarized one by one, then merged.

        Each part's summary is cached as it's made (style "part", keyed by the chunk size, which decides the
        parts), so a failed merge or an interrupted run redoes only what's missing; both styles share them.
        """
        parts = _pieces(chapter_text(self.pages, self.chapters[i]), config.BOOK_CHUNK_CHARS)
        cached = db.get_doc_summaries(self.sha, "part", self.backend, self.model)
        part_summaries = []
        for k, part in enumerate(parts, 1):
            key = f"{i}:p{k}/{config.BOOK_CHUNK_CHARS}"
            if key in cached:
                part_summaries.append(cached[key]["summary"])
                continue
            self.status(f"🧠 {self.llm}: a long chapter, part {k} of {len(parts)}…",
                        self._eta(sum(map(len, parts[k - 1:])), len(parts) - k + 2))
            body = f'<chapter index="0" title="{html.escape(self.chapters[i]["title"])} (part {k})">\n{part}\n</chapter>'
            answer = self._ask(summarize.CHAPTERS_PROMPT.format(length=_length("full", budget)),
                               f"{_material(self.doc)}\n{body}", summarize.CHAPTERS_SCHEMA)
            summary = next((e["summary"] for e in answer["chapters"]), "")
            db.save_doc_summary(self.sha, key, "part", self.backend, self.model, {"summary": summary})
            part_summaries.append(summary)
        self.status(f"🧠 {self.llm}: a long chapter, putting the parts together…", self._eta(0))
        joined = "\n\n".join(f"<chapter part={k}>\n{s}\n</chapter>" for k, s in enumerate(part_summaries, 1))
        answer = self._ask(summarize.COMBINE_PROMPT.format(length=_length(style, budget)), joined,
                           summarize.TEXT_SCHEMA)
        return answer["summary"]

    def book(self) -> dict:
        """The whole-book summary: {title, author, summary}, from the cache where possible.

        One call when the text fits; otherwise from all chapters' short summaries (which are cached too, so
        "all chapters, short" is free afterwards).

        Raises:
            summarize.SummaryError: The AI failed.
        """
        if cached := db.get_doc_summaries(self.sha, "full", self.backend, self.model).get("book"):
            return cached
        text = "\n".join(self.pages[i] for i in sorted(self.pages)).strip()
        if len(text) <= config.BOOK_CHUNK_CHARS:
            self.status(f"🧠 Summarizing with {self.llm}…", self._eta(len(text)))
            answer = self._ask(summarize.BOOK_PROMPT, f"{_material(self.doc)}\n<document>\n{text}\n</document>",
                               summarize.BOOK_SCHEMA)
        else:
            shorts = self.chapters_summaries(list(range(len(self.chapters))), "short")
            self.status(f"🧠 {self.llm}: putting the book together…", self._eta(sum(map(len, shorts.values()))))
            body = "\n".join(f'<chapter index="{i}" title="{html.escape(self.chapters[i]["title"])}">\n{s}\n</chapter>'
                             for i, s in shorts.items())
            answer = self._ask(summarize.BOOK_FROM_CHAPTERS_PROMPT, f"{_material(self.doc)}\n{body}",
                               summarize.BOOK_SCHEMA)
        result = {k: answer[k] for k in ("title", "author", "summary")}
        db.save_doc_summary(self.sha, "book", "full", self.backend, self.model, result)
        return result
