"""Reads an uploaded document in the sandbox and keeps its text: the bot-side half of docparse.

The file is parsed by `python -m summarizer.docparse` inside the sandbox (see sandbox.py); only its JSON
result comes back. Pages and chapters are stored by the file's SHA-256, so a document is read once however
often (and by whomever) it is summarized.
"""
import hashlib
import json
import logging
from pathlib import Path

from . import config, db, proc, sandbox

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


class DocumentError(RuntimeError):
    """The document can't be used. str() is a message for the user; `detail` is for the admins."""

    def __init__(self, message: str, detail: str | None = None):
        """Stores the user message and an optional technical detail."""
        super().__init__(message)
        self.detail = detail


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


def store(digest: str, name: str, parsed: dict) -> None:
    """Saves a parsed document: metadata and chapters in documents, the text in document_pages."""
    db.save_document(digest, name=name[:200], format=parsed["format"], pages=len(parsed["pages"]),
                     title=parsed.get("title") or None, author=parsed.get("author") or None,
                     chapters=parsed["chapters"], text_source="text", status="done", error=None)
    db.save_pages(digest, dict(enumerate(parsed["pages"])), "text")
