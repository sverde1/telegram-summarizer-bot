"""Turning results into Telegram messages: summaries, documents, footers, and the time/size formats."""
import html
import math

from summarizer import documents, ocr, pipeline, stats, summarize, units


TG_LIMIT = 4096  # Telegram's maximum message length in characters


def fmt_until(seconds: float) -> str:
    """Coarse time until something, rounded up: "25 min" under an hour, else "3 h"."""
    if seconds < 3600:
        return f"{max(1, math.ceil(seconds / 60))} min"
    return f"{math.ceil(seconds / 3600)} h"


def fmt_size(n: int) -> str:
    """File size as "850 KB" or "2.3 MB"."""
    return f"{n / 1024 ** 2:.1f} MB" if n >= 1024 ** 2 else f"{max(1, round(n / 1024))} KB"


def _secs(sec: float) -> str:
    """Formats a duration for the footer: "42 s" under a minute, "m:ss" from there on.

    Args:
        sec: Duration in seconds.

    Returns:
        The formatted duration.
    """
    sec = round(sec)
    return f"{sec} s" if sec < 60 else f"{sec // 60}:{sec % 60:02d}"


def details(r: pipeline.Result, waited: float = 0, reveal_cache: bool = True) -> str:
    """Builds the footer: how long each step took and what was used (transcript, frames, LLM).

    Args:
        r: The pipeline result.
        waited: Seconds the job waited in the queue; shown when it's 5 s or more.
        reveal_cache: Whether this user may learn the result came from the cache. A result with replay
            steps (see pipeline.plan_replay) shows them, like a fresh run's timings, so the footer matches how
            long the user actually waited.

    Returns:
        One or two lines of plain text (the caller HTML-escapes it).
    """
    stats = (r.summary or {}).get("_stats") or {}
    if r.replay_steps:
        steps = " · ".join(f"{name} {_secs(sec)}" for name, sec in r.replay_steps)
        timing = f"⏱ {_secs(r.replay_total)} total: {steps}"
        if waited >= 5:
            timing += f" (+ {_secs(waited)} waiting in queue)"
    elif r.cached and not reveal_cache:
        timing = ""  # never "from cache" (pipeline sets replay steps for these users; this is a backstop)
    elif r.cached:
        timing = "⚡ from cache" + (f" (first run took {_secs(stats['total'])})" if stats else "")
    elif stats:
        steps = " · ".join(f"{name} {_secs(sec)}" for name, sec in stats["steps"])
        timing = f"⏱ {_secs(stats['total'])} total: {steps}"
        if waited >= 5:
            timing += f" (+ {_secs(waited)} waiting in queue)"
    else:
        timing = ""
    source = {"captions": "YouTube captions", "tiktok-webvtt": "TikTok captions", "none": "none"}.get(
        r.transcript_source, r.transcript_source.replace("whisper-", "Whisper "))
    if r.transcript_source == "none":
        used = "📝 no speech found" if not r.meta.get("is_carousel") else "📝 photo post, no transcript"
    else:
        used = f"📝 transcript: {source}" + (f" ({r.language})" if r.language else "")
    if r.frames_used:
        used += " · 🖼 slides" if r.meta.get("is_carousel") else " · 🎞 video frames"
    # The model saved with the summary, not the current default: a cached summary may be from another model.
    used += f" · 🧠 {stats.get('llm') or summarize.llm_label()}"
    return "\n".join(filter(None, [timing, used]))


# Longest a model-written field may be. The prompt asks for much less; these only stop a broken or
# prompt-injected answer from turning into a flood of messages.
FIELD_LIMITS = {"title": 300, "clickbait_answer": 1000, "summary": 4000}


def _cap(text: str, limit: int) -> str:
    """Shortens text to at most `limit` characters, marking the cut with "…"."""
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def _escaped_pieces(raw_line: str) -> list[str]:
    """HTML-escapes one line of text, cut into pieces that each fit a message.

    The cut is made on the raw text, so it can never land inside an entity like `&amp;` (which Telegram
    rejects); pieces are sized by their escaped length, which can be up to 6x the raw length.
    """
    pieces, current, size = [], [], 0
    for ch in raw_line:
        esc = html.escape(ch)
        if size + len(esc) > TG_LIMIT:
            pieces.append("".join(current))
            current, size = [], 0
        current.append(esc)
        size += len(esc)
    pieces.append("".join(current))
    return pieces


def _pack(pieces: list[str]) -> list[str]:
    """Joins message pieces with newlines into as few messages as possible, each at most TG_LIMIT long.

    Every piece is complete HTML on its own (escaped text or a whole tag pair), so splitting between pieces
    keeps each message valid.
    """
    chunks, current = [], ""
    for piece in pieces:
        candidate = f"{current}\n{piece}" if current else piece
        if len(candidate) > TG_LIMIT and current:
            chunks.append(current)
            candidate = piece
        current = candidate
    if current.strip():
        chunks.append(current)
    return chunks


def render(r: pipeline.Result, waited: float = 0, reveal_cache: bool = True,
           units_: tuple[str, str] = ("metric", "c")) -> list[str]:
    """Builds the reply in the Title / Clickbait answer / Summary layout, split to fit Telegram.

    All model and video text is HTML-escaped: messages are sent with parse_mode=HTML, and titles or
    summaries containing `<` or `&` would otherwise break parsing (or inject markup). Model fields are capped
    (FIELD_LIMITS) and messages are packed by their escaped length, so no message exceeds Telegram's limit.

    Args:
        r: The pipeline result (must have a summary).
        waited: Seconds the job waited in the queue (for the footer).
        reveal_cache: Whether this user may learn the result came from the cache (see `details`).

    Returns:
        One or more HTML messages, each at most TG_LIMIT characters.
    """
    s = r.summary
    title = _cap(s.get("title") or r.meta.get("title", ""), FIELD_LIMITS["title"])
    if s.get("is_clickbait") and s.get("clickbait_answer"):
        answer = _cap(units.convert(s["clickbait_answer"], *units_), FIELD_LIMITS["clickbait_answer"])
    else:
        answer = "✅ Not clickbait - the title matches the content."
    footer = _cap(details(r, waited, reveal_cache), 1000) + "\n" + r.url
    # A recording someone sent has no published title or thumbnail: nothing to call clickbait.
    clickbait = [] if r.platform == "file" else ["<b>Clickbait answer:</b>", *_text(answer), ""]
    pieces = ["<b>Title:</b>", *_text(title), "", *clickbait,
              "<b>Summary:</b>", *_text(_cap(units.convert(s.get("summary", ""), *units_), FIELD_LIMITS["summary"])),
              ""]
    # The footer is one italic piece: an <i> split across two messages would break both.
    pieces.append(f"<i>{html.escape(_cap(footer, 1500))}</i>")
    return _pack(pieces)


def _text(raw: str) -> list[str]:
    """Escaped message pieces for a block of text, one or more per line."""
    return [piece for line in raw.split("\n") for piece in _escaped_pieces(line)]


def _doc_details(r: documents.DocResult, waited: float, reveal_cache: bool) -> str:
    """The footer of a document summary: timings (like a fresh run for first-time requesters), the source."""
    if r.replay_steps:
        timing = f"⏱ {_secs(r.replay_total)} total: " + " · ".join(f"{n} {_secs(s)}" for n, s in r.replay_steps)
    elif r.cached:
        timing = "⚡ from cache" if reveal_cache else ""
    elif r.steps:
        timing = f"⏱ {_secs(r.total)} total: " + " · ".join(f"{n} {_secs(s)}" for n, s in r.steps)
    else:
        timing = ""
    if timing and waited >= 5:
        timing += f" (+ {_secs(waited)} waiting in queue)"
    fmt = (r.doc.get("format") or "").upper()
    pages = f"{r.doc.get('pages')} pages" if fmt == "PDF" else f"~{r.doc.get('pages')} pages"
    source = r.doc.get("text_source") or ""
    if source.startswith("ocr-"):
        engine_name = {"tesseract": "Tesseract", "rapidocr": "RapidOCR"}.get(source[4:], source[4:])
        fmt += f" · 🔍 OCR ({engine_name}, {ocr_names(r.doc.get('language'))})"
    used = f"📄 {r.name[:80]} · {fmt} · {pages} · 🧠 {r.llm}"
    return "\n".join(filter(None, [timing, used]))


def render_document(r: documents.DocResult, waited: float = 0, reveal_cache: bool = True,
                    units_: tuple[str, str] = ("metric", "c")) -> list[str]:
    """Builds the messages for a document summary, each at most TG_LIMIT characters.

    Whole book: Title / Author / Summary. All chapters short: one block per chapter, packed into as few
    messages as fit. One per message: a message per chapter. One chapter: its title and summary. The footer
    goes on the last message. Everything from the file or the model is escaped and capped.
    """
    footer = f"<i>{html.escape(_cap(_doc_details(r, waited, reveal_cache), 1500))}</i>"
    if r.kind == "book":
        b = r.book or {}
        pieces = ["<b>Title:</b>", *_text(_cap(b.get("title") or r.name, FIELD_LIMITS["title"]))]
        if b.get("author"):
            pieces += ["", "<b>Author:</b>", *_text(_cap(b["author"], FIELD_LIMITS["title"]))]
        pieces += ["", "<b>Summary:</b>", *_text(_cap(units.convert(b.get("summary", ""), *units_),
                                                    FIELD_LIMITS["summary"])), "", footer]
        return _pack(pieces)
    blocks = [[f"<b>{html.escape(_cap(title, 200))}</b>",
               *_text(_cap(units.convert(summary, *units_), FIELD_LIMITS["summary"]))]
              for _, title, summary in r.chapters]
    if r.kind == "short":
        pieces = [f"<b>📑 {html.escape(r.name[:80])}</b>", ""]
        for block in blocks:
            pieces += [*block, ""]
        return _pack(pieces + [footer])
    if r.kind == "each" and len(blocks) > 1:
        n = len(blocks)
        out = []
        for k, block in enumerate(blocks, 1):
            head = [f"<b>{k}/{n}</b> " + block[0], *block[1:]]
            out += _pack(head + (["", footer] if k == n else []))
        return out
    return _pack([piece for block in blocks for piece in block] + ["", footer])


def fmt_eta(sec: float) -> str:
    """Formats the remaining-time estimate, deliberately coarse so it reads as an estimate.

    Under a minute it's rounded *up* to 5 s (an optimistic ETA that keeps running out is worse than a
    slightly pessimistic one); under 10 min to half minutes; beyond that to whole minutes.

    Args:
        sec: Estimated seconds left.

    Returns:
        E.g. "15 s", "2.5 min", "12 min".
    """
    sec = max(int(sec), 0)
    if sec < 60:
        return f"{max(-(-sec // 5) * 5, 5)} s"  # round up to 5 s
    return f"{round(sec / 30) / 2:g} min" if sec < 600 else f"{round(sec / 60)} min"


def ocr_names(langs: str | None) -> str:
    """Names of a Tesseract language string, e.g. "slv+eng" -> "Slovenian, English"."""
    return ocr.names((langs or "").split("+")) or "unknown"
