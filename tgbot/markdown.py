"""📄 Download: a summary as a Markdown file, to take to ChatGPT or any other chat.

Videos and recordings: the summary (in the reader's units) and the timestamped transcript. Books and
documents: the full text only, by the owner's choice (the summaries are already in the chat).
"""
import re

from summarizer import books, db, units

MAX_BYTES = 50 * 1024 ** 2  # Telegram lets bots send documents up to 50 MB


def filename(title: str) -> str:
    """A safe file name for a title: letters, digits, spaces, dots and dashes, at most 60 characters."""
    name = re.sub(r"[^\w .-]+", "", title).strip(" .")[:60].strip(" .")
    return f"{name or 'summary'}.md"


def video(delivered: dict, link: str, transcript: str, units_: tuple[str, str]) -> str:
    """A video's or recording's file: title, link (or label), summary, transcript.

    Args:
        delivered: Its `delivered` row.
        link: The video's URL, or a recording's label.
        transcript: The stored transcript ("" when there is none).
        units_: The reader's (unit system, temperature) for the summary.
    """
    summary = units.convert(delivered["text"], *units_)  # display-only: converted here, never stored
    body = transcript.strip() or "No transcript (a photo post, or a video without speech)."
    return f"# {delivered['title']}\n\n{link}\n\n## Summary\n\n{summary}\n\n## Transcript\n\n{body}\n"


def document(title: str, doc: dict, pages: dict[int, str]) -> str:
    """A book's or document's file: its full text, with its chapters as headings when it has several."""
    chapters = doc.get("chapters") or []
    if len(chapters) > 1:
        parts = [f"## {ch['title']}\n\n{books.chapter_text(pages, ch)}" for ch in chapters]
    else:
        parts = ["\n\n".join(pages[i].strip() for i in sorted(pages))]
    return f"# {title}\n\n" + "\n\n".join(parts) + "\n"


def build(request_id: int, units_: tuple[str, str]) -> tuple[str, bytes] | None:
    """The file for a delivered summary (blocking: run it in a thread, a book's text can be megabytes).

    Returns:
        (file name, UTF-8 bytes), or None when the summary or its text is no longer stored.
    """
    delivered, req = db.get_delivered(request_id), db.get_request(request_id)
    if not delivered or not req or delivered["kind"] == "ask":
        return None
    if delivered["kind"] == "document":
        doc, pages = db.get_document(req["video_id"] or ""), db.get_pages(req["video_id"] or "")
        if not doc or not pages:
            return None
        text = document(delivered["title"], doc, pages)
    else:
        row = db.get_video(req["platform"] or "", req["video_id"] or "") or {}
        # A recording's label comes from its own request: the shared videos row never has the file name.
        link = req["url"] if delivered["kind"] == "file" else (row.get("url") or req["url"])
        text = video(delivered, link, row.get("transcript") or "", units_)
    return filename(delivered["title"]), text.encode()
