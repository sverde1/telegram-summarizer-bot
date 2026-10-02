"""Reads an uploaded document in the sandbox and keeps its text: the bot-side half of docparse.

The file is parsed by `python -m summarizer.docparse` inside the sandbox (see sandbox.py); only its JSON
result comes back. Pages and chapters are stored by the file's SHA-256, so a document is read once however
often (and by whomever) it is summarized.
"""
import hashlib
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from . import books, config, db, docparse, ocr, pipeline, proc, sandbox, summarize

log = logging.getLogger(__name__)

PARSE_TIMEOUT = 600  # seconds; a 2000-page PDF takes about a minute

# What the user sees for each docparse error code.
MESSAGES = {
    "unsupported": "⚠️ I can read PDF, EPUB, DOCX and TXT files. Convert it to PDF or EPUB and send it again.",
    "encrypted": "⚠️ This PDF is password-protected. Send a version without the password.",
    "drm": "⚠️ This e-book is copy-protected (DRM), so I can't read it.",
    "too_many_pages": f"⚠️ This document has too many pages (I read up to {config.MAX_DOC_PAGES}).",
    "too_large": "⚠️ This document has too much text for me to summarize.",
    "zip_bomb": "⚠️ This file looks damaged, so I can't read it.",
    "broken": "⚠️ This file looks damaged, so I can't read it.",
}


EMPTY = "⚠️ I couldn't find any readable text in this file."
MIN_LETTERS = 50          # less than this in the whole file is "no readable text"
SCAN_PAGE_LETTERS = 20    # a PDF page with fewer letters has no text layer (a short real page still has more)
SCAN_SHARE = 0.5          # …and a PDF where at least this share of pages is like that is a scan


class DocumentError(pipeline.PipelineError):
    """The document can't be used. str() is a message for the user; `detail` is for the admins."""


class NeedsOcr(Exception):
    """A scanned document: the user must confirm OCR first (it takes long and has its own daily limit).

    Attributes:
        pages: How many pages need OCR.
        seconds: Estimated OCR time.
        language: The OCR language(s) chosen from the sample, e.g. "slv".
    """

    def __init__(self, pages: int, seconds: float, language: str):
        """Stores what the confirmation message shows."""
        super().__init__(f"{pages} pages need OCR")
        self.pages, self.seconds, self.language = pages, seconds, language


@dataclass
class DocResult:
    """A finished document request, for the bot to render.

    Attributes:
        kind: book | short | each | chapter | pick (pick: only the chapter list is wanted).
        name: The file name as uploaded.
        doc: The documents row (format, pages, chapters, text_source, ...).
        book: {title, author, summary} for kind "book".
        chapters: [(chapter index, title, summary)] for the chapter kinds.
        steps: (label, seconds) of the work done, for the footer.
        total: Seconds the whole job took.
        llm: Label of the model that wrote the summaries.
        cached: Whether nothing had to be done (text and summaries came from the database).
        replay_steps / replay_total: As for videos (see pipeline.Result): what a first-time requester is
            shown instead of an instant answer.
    """
    kind: str
    name: str
    doc: dict
    book: dict | None = None
    chapters: list[tuple[int, str, str]] = field(default_factory=list)
    steps: list[tuple[str, float]] = field(default_factory=list)
    total: float = 0.0
    llm: str = ""
    cached: bool = False
    replay_steps: list[tuple[str, float]] | None = None
    replay_total: float = 0.0
    platform: str = "document"  # with video_id, what db.user_saw_video looks for
    video_id: str = ""
    summary: dict | None = None  # videos only; here for the code that handles both

    @property
    def head(self) -> str:
        """First line of the status message while replaying."""
        return f"📄 {self.name[:80]} ({self.doc.get('pages') or '?'} pages)"


def sha256(path: Path) -> str:
    """Hex SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def parse(path: Path, workdir: Path) -> dict:
    """Parses a file in the sandbox.

    Args:
        path: The downloaded file; must be inside `workdir`.
        workdir: The job's directory (the only one the sandbox can write).

    Returns:
        docparse's result: format, title, author, pages, chapters (and letters per page for PDFs).

    Raises:
        DocumentError: The file can't be read (message for the user).
        proc.ProcCancelled: The job was cancelled.
    """
    out = workdir / "parsed.json"
    cmd = sandbox.command(workdir, ["python", "-m", "summarizer.docparse", f"/job/{path.relative_to(workdir)}",
                                    "/job/parsed.json", str(config.MAX_DOC_PAGES), str(config.MAX_DOC_CHARS)])
    try:
        run = proc.run(cmd, timeout=PARSE_TIMEOUT)
    except proc.ProcTimeout:
        raise DocumentError(MESSAGES["broken"], "parsing timed out")
    if not out.exists():
        # Killed by the memory limit, or the sandbox itself failed: either way, nothing usable.
        raise DocumentError(MESSAGES["broken"], f"parser exited {run.returncode}: {run.stderr[-500:]}")
    result = json.loads(out.read_text())
    if code := result.get("error"):
        raise DocumentError(MESSAGES.get(code, MESSAGES["broken"]), f"{code}: {result.get('detail', '')}")
    return result


def read_seconds(pages: int) -> float:
    """Rough time to read (download and parse) a document, for replays and pacing."""
    return 2 + 0.02 * pages


def _letters(text: str) -> int:
    """Number of letters in a text."""
    return sum(c.isalpha() for c in text)


def _is_scan(parsed: dict) -> bool:
    """Whether a PDF is mostly pictures of pages: at least SCAN_SHARE of its pages have no text layer."""
    letters = parsed.get("letters")
    return bool(letters) and sum(1 for n in letters if n < SCAN_PAGE_LETTERS) >= SCAN_SHARE * len(letters)


def _check_text(pages) -> None:
    """Refuses a document with no readable text at all.

    Raises:
        DocumentError: EMPTY.
    """
    if sum(_letters(p) for p in pages) < MIN_LETTERS:
        raise DocumentError(EMPTY)


def _ocr_pages(digest: str) -> list[int]:
    """Pages of a scanned document still without text (neither a text layer nor OCR yet)."""
    sources, pages = db.get_page_sources(digest), db.get_pages(digest)
    return [p for p in sorted(pages) if sources.get(p) == "text" and _letters(pages[p]) < SCAN_PAGE_LETTERS]


def _ocr(digest: str, doc: dict, path: Path | None, workdir: Path, st: "pipeline.Status", request_id: int,
         confirmed: bool) -> list[tuple[str, float]]:
    """Makes a scanned document's text: asks for confirmation first, then OCRs the pages without text.

    Returns:
        The steps done, for the footer.

    Raises:
        NeedsOcr: Not confirmed yet (the bot asks the user).
        DocumentError: OCR isn't available, the language isn't supported, or no text came out.
    """
    if not ocr.available():
        raise DocumentError(ocr.NO_ENGINE, f"OCR engine {ocr.engine()} not available")
    if path is None:
        raise DocumentError("⚠️ Please send the file again.", "scan without the file")
    need = _ocr_pages(digest)
    steps = []
    if need and not doc.get("language"):  # the language is checked once per document
        st.show("🔍 Checking the scan's language…")
        t = time.monotonic()
        try:
            langs = ocr.choose_languages(path, need, workdir)
        except ocr.UnsupportedLanguage as e:
            raise DocumentError(str(e), "unsupported OCR language")
        db.save_document(digest, language=langs)
        doc["language"] = langs
        steps.append(("language check", time.monotonic() - t))
    if need and not confirmed:
        raise NeedsOcr(len(need), ocr.estimate(len(need)), doc["language"])
    if need:
        db.update_request(request_id, ocr=1)  # counts toward the OCR limit from the moment it starts
        t = time.monotonic()
        ocr.run(path, digest, need, doc["language"], workdir, st.show)
        steps.append(("OCR", time.monotonic() - t))
    pages = db.get_pages(digest)
    ordered = [pages[i] for i in sorted(pages)]
    _check_text(ordered)
    fields = {"status": "done", "text_source": f"ocr-{ocr.engine()}", "error": None}
    if not doc.get("toc"):  # no bookmarks: now that there's text, look for chapter headings
        fields["chapters"] = docparse.pdf_chapters(ordered)
    db.save_document(digest, **fields)
    return steps


def run(upload: dict, mode: str, chapter: int | None, path: Path | None, workdir: Path,
        progress: Callable[..., None], *, backend: str | None, model: str | None, request_id: int,
        hide_cache_from: int | None, ocr_confirmed: bool = False) -> DocResult:
    """Reads an uploaded document (unless already read) and summarizes it as the user chose.

    Args:
        upload: The uploads row.
        mode: "whole", "short" (all chapters, short), "each" (all chapters, full), "pick".
        chapter: For "pick": the chapter to summarize; None to only produce the chapter list.
        path: The downloaded file, or None when the document's text is already stored.
        workdir: The job's directory (the sandbox's only writable place).
        progress: Callback (text, eta) for the status message.
        backend: The user's backend; None = LLM_BACKEND.
        model: The user's model; None = that backend's default.
        request_id: The requests row, updated with the document and status.
        hide_cache_from: A non-admin requester who mustn't learn that someone else sent this file before.
        ocr_confirmed: The user confirmed OCR for a scan (see NeedsOcr).

    Returns:
        The result to render.

    Raises:
        DocumentError: The file can't be read or summarized (message for the user).
        NeedsOcr: A scan whose OCR the user hasn't confirmed yet.
        proc.ProcCancelled: The job was cancelled.
    """
    t0 = time.time()
    backend = backend or config.LLM_BACKEND
    model = model or summarize.default_model(backend)
    st = pipeline.Status(progress)
    st.head = f"📄 {upload['name'][:80]}"
    steps: list[tuple[str, float]] = []
    digest = upload.get("sha256") or (sha256(path) if path else None)
    if not digest:
        raise DocumentError(MESSAGES["broken"], "neither a stored document nor a file")
    db.set_upload_sha(upload["id"], digest)
    db.update_request(request_id, platform="document", video_id=digest, status="processing")
    hide = bool(hide_cache_from) and not db.user_saw_video(hide_cache_from, "document", digest, request_id)
    doc = db.get_document(digest)
    # A scan read before but not OCRed yet ("needs-ocr") isn't parsed again: its pages are stored.
    read_now = not doc or doc["status"] not in ("done", "needs-ocr")
    if read_now:
        if path is None:
            raise DocumentError("⚠️ Please send the file again.", "document text missing")
        st.show("📄 Reading the file…")
        t = time.monotonic()
        db.save_document(digest, name=upload["name"][:200], status="processing")
        try:
            parsed = parse(path, workdir)
            scan = _is_scan(parsed)
            if not scan:
                _check_text(parsed["pages"])
        except DocumentError as e:
            db.save_document(digest, status="failed", error=(e.detail or str(e))[:500])
            raise
        store(digest, upload["name"], parsed, scan)
        steps.append(("reading the file", time.monotonic() - t))
        doc = db.get_document(digest)
    if doc["status"] == "needs-ocr":
        try:
            steps += _ocr(digest, doc, path, workdir, st, request_id, ocr_confirmed)
        except DocumentError as e:
            if str(e) == EMPTY:
                db.save_document(digest, status="failed", error="no text after OCR")
            raise
        doc = db.get_document(digest)
        read_now = True
    st.head = f"📄 {upload['name'][:80]} ({doc['pages']} pages)"
    st.ok(f"✅ {len(doc['chapters'])} chapters" if len(doc["chapters"]) > 1 else "✅ File read")

    b = books.Books(digest, backend, model, st.show)
    kind = {"whole": "book", "short": "short", "each": "each"}.get(mode) or ("pick" if chapter is None else "chapter")
    result = DocResult(kind, upload["name"], doc, video_id=digest, llm=b.llm)
    if kind == "chapter" and not 0 <= chapter < len(doc["chapters"]):
        raise DocumentError("⚠️ That chapter doesn't exist any more. Please pick again.", f"chapter {chapter}")
    summary_cached = _summaries_cached(b, result.kind, chapter)
    if hide and not read_now and not summary_cached:
        # The text was read before, for someone else: show the reading step a fresh run takes (paced like a
        # cached answer), or an instant "file read" would tell them.
        pause = min(read_seconds(doc["pages"]) * pipeline.REPLAY_SHARE, pipeline.REPLAY_MAX)
        st.show("📄 Reading the file…")
        pipeline._pause(pause)
        steps.append(("reading the file", pause))
    try:
        if result.kind == "book":
            result.book = b.book()
        elif result.kind in ("short", "each"):
            all_ = list(range(len(doc["chapters"])))
            got = b.chapters_summaries(all_, "short" if result.kind == "short" else "full")
            result.chapters = [(i, doc["chapters"][i]["title"], got[i]) for i in all_]
        elif result.kind == "chapter":
            got = b.chapters_summaries([chapter], "full")
            result.chapters = [(chapter, doc["chapters"][chapter]["title"], got[chapter])]
    except summarize.SummaryError as e:
        raise DocumentError(str(e), e.detail)
    if b.steps:
        steps.append(("summary", sum(sec for _, sec in b.steps)))
    result.steps, result.llm, result.total = steps, b.llm, time.time() - t0
    result.cached = not read_now and not b.steps and result.kind != "pick"
    if result.cached and hide:
        _plan_replay(result, backend)
    return result


def _summaries_cached(b: "books.Books", kind: str, chapter: int | None) -> bool:
    """Whether everything this request needs from the AI is cached already."""
    if kind == "pick":
        return False
    if kind == "book":
        return "book" in db.get_doc_summaries(b.sha, "full", b.backend, b.model)
    style = "short" if kind == "short" else "full"
    have = db.get_doc_summaries(b.sha, style, b.backend, b.model)
    needed = [chapter] if kind == "chapter" else range(len(b.chapters))
    return all(str(i) in have for i in needed)


def _plan_replay(result: DocResult, backend: str) -> None:
    """A cached answer for a first-time requester: the steps a fresh run shows, at half their time (≤ 2 min)."""
    pages = result.doc.get("pages") or 0
    chars = sum(len(t) for t in db.get_pages(result.video_id).values())
    calls = 1 if result.kind in ("book", "chapter") else max(1, -(-len(result.chapters) // books.PER_CALL[
        "short" if result.kind == "short" else "full"]))
    steps = [("reading the file", read_seconds(pages)),
             ("summary", calls * pipeline._eta_llm(min(chars, config.BOOK_CHUNK_CHARS) / calls, 0, backend))]
    if result.kind == "chapter":
        ch = result.doc["chapters"][result.chapters[0][0]]
        steps[1] = ("summary", pipeline._eta_llm(sum(len(db.get_pages(result.video_id).get(i, ""))
                                                     for i in range(ch["start"], ch["end"])), 0, backend))
    total = sum(sec for _, sec in steps)
    delay = min(total * pipeline.REPLAY_SHARE, pipeline.REPLAY_MAX)
    result.replay_steps = [(name, sec * delay / total) for name, sec in steps]
    result.replay_total = delay


def store(digest: str, name: str, parsed: dict, scan: bool = False) -> None:
    """Saves a parsed document: metadata and chapters in documents, the text in document_pages.

    A scan is saved as "needs-ocr": its pages are there (mostly empty), OCR fills them in later.
    """
    db.save_document(digest, name=name[:200], format=parsed["format"], pages=len(parsed["pages"]),
                     title=parsed.get("title") or None, author=parsed.get("author") or None,
                     chapters=parsed["chapters"], text_source="text", toc=int(bool(parsed.get("toc"))),
                     status="needs-ocr" if scan else "done", error=None)
    db.save_pages(digest, dict(enumerate(parsed["pages"])), "text")
