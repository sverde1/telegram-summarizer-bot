"""💬 Ask: answering a reader's follow-up question about a summary they received.

Each question is one fresh AI call (summarize.ask: same sandbox and tool lockdown as every call) with the
material the summary came from: a video's or recording's transcript, or a book's text. The whole prompt stays
within ASK_MAX_CHARS; a long summary is cut first, then the material, and a book too long to send falls back to
its stored summaries plus the chapters the question names. Answers are never shared between users.
Blocking: runs in a worker thread.
"""
import logging
import re
from dataclasses import dataclass

from . import books, config, db, pipeline, summarize
from .results import JobResult

log = logging.getLogger(__name__)

TOO_OLD = "This summary is too old for questions; ask for it again."
# The reply to an unrelated request, by what the summary was of (the prompt asks for exactly this sentence).
OFF_TOPIC = {"video": "I can only answer questions about topics related to the video you sent.",
             "file": "I can only answer questions about topics related to the recording you sent.",
             "document": "I can only answer questions about topics related to the book or document you sent."}
HISTORY = 3  # earlier questions and answers about the same summary sent along (this user's only)
_CHAPTER_NUMBER = re.compile(r"\b(?:chapter|chap\.?|ch\.?|poglavje|poglavju|poglavja)\s*(\d{1,3})\b", re.I)


@dataclass(kw_only=True)
class AskResult(JobResult):
    """An answer to a follow-up question.

    Attributes:
        question: The reader's question.
        answer: The AI's answer (before unit conversion).
        parent_id: The summary request it is about.
        llm: Label of the model that answered.
    """
    question: str
    answer: str
    parent_id: int
    llm: str = ""

    def head(self) -> str:
        """The status line: the question."""
        return f"💬 {self.question[:60]}"

    def llm_label(self) -> str:
        """The model that answered."""
        return self.llm


def _cut(text: str, limit: int, what: str) -> str:
    """Text within limit characters, with a note for the AI when it was cut."""
    if len(text) <= limit:
        return text
    return text[:max(0, limit - 60)] + f"\n[… the rest of the {what} is left out: too long]"


def _video_material(req: dict, budget: int) -> str:
    """A video's or recording's transcript, within the budget."""
    row = db.get_video(req["platform"] or "", req["video_id"] or "") or {}
    if not (row.get("transcript") or "").strip():
        # Photo posts and silent videos: the summary came from what was shown, which the AI can't see now.
        return "<transcript>(no transcript; the summary may rely on what was shown on screen)</transcript>"
    return f"<transcript>\n{_cut(row['transcript'], budget - 30, 'transcript')}\n</transcript>"


def named_chapters(question: str, chapters: list[dict]) -> list[int]:
    """The chapter indexes a question names: by number ("chapter 7") or by a title of at least 4 characters."""
    found = [int(n) - 1 for n in _CHAPTER_NUMBER.findall(question)]
    q = question.lower()
    found += [i for i, ch in enumerate(chapters) if len(ch["title"].strip()) >= 4 and ch["title"].lower() in q]
    return [i for i in dict.fromkeys(found) if 0 <= i < len(chapters)]


def _book_material(req: dict, delivered: dict, question: str, budget: int) -> tuple[str, bool]:
    """A book's full text when it fits, else its stored summaries plus the chapters the question names.

    Returns:
        (material, whether the full text was sent).
    """
    sha = req["video_id"] or ""
    doc, pages = db.get_document(sha), db.get_pages(sha)
    if not doc or not pages:
        raise pipeline.PipelineError(TOO_OLD, detail=f"document {sha[:12]} has no stored text")
    full = "\n".join(pages[i] for i in sorted(pages))
    if len(full) + 30 <= budget:
        return f"<document>\n{full}\n</document>", True
    chapters = doc.get("chapters") or []
    # Summaries by the model that wrote the delivered one (they're cached per model).
    backend, model = delivered["backend"] or "", delivered["model"] or ""
    parts = []
    if book := db.get_doc_summaries(sha, "full", backend, model).get("book"):
        parts.append(f"<chapter_summary title=\"whole book\">\n{book.get('summary', '')}\n</chapter_summary>")
    by_part = db.get_doc_summaries(sha, "full", backend, model) | db.get_doc_summaries(sha, "short", backend, model)
    for i, ch in enumerate(chapters):
        if (s := by_part.get(str(i))) and s.get("summary"):
            parts.append(f"<chapter_summary index=\"{i + 1}\" title=\"{ch['title'][:100]}\">\n{s['summary']}\n"
                         "</chapter_summary>")
    material = "\n".join(parts)
    for i in named_chapters(question, chapters):
        room = budget - len(material) - 80
        if room < 1000:
            break
        text = _cut(books.chapter_text(pages, chapters[i]), room, "chapter")
        material += f"\n<chapter index=\"{i + 1}\" title=\"{chapters[i]['title'][:100]}\">\n{text}\n</chapter>"
    note = ("(The full text is too long to send: these are its chapter summaries, plus the full text of any "
            "chapter the question names. Say so if the answer needs more detail than that.)\n")
    return note + _cut(material, budget - len(note), "summaries"), False


def prompt(parent_id: int, question: str, user_id: int) -> str:
    """The material, summary, this user's earlier Q&As about it and the question, within ASK_MAX_CHARS.

    Raises:
        pipeline.PipelineError: The summary or its material is no longer stored.
    """
    delivered, req = db.get_delivered(parent_id), db.get_request(parent_id)
    if not delivered or not req or delivered["kind"] == "ask":
        raise pipeline.PipelineError(TOO_OLD, detail=f"request {parent_id} has no delivered summary")
    history = "".join(f"<earlier_answer question=\"{h['title'][:300]}\">\n{h['text']}\n</earlier_answer>\n"
                      for h in db.ask_history(parent_id, user_id, HISTORY))
    budget = config.ASK_MAX_CHARS - len(question) - len(history) - 200
    # A long summary ("each" chapter of a big book) is cut before the material: the material holds the facts.
    summary = _cut(delivered["text"], max(budget // 3, 2000), "summary")
    rest = budget - len(summary)
    if delivered["kind"] == "document":
        material, _ = _book_material(req, delivered, question, rest)
    else:
        material = _video_material(req, rest)
    return (f"<summary title=\"{delivered['title'][:300]}\">\n{summary}\n</summary>\n{history}{material}\n"
            f"<question>\n{question}\n</question>")


def off_topic_rule(kind: str) -> str:
    """The last line of the instructions: the exact reply to an unrelated request, for this kind of summary."""
    sentence = OFF_TOPIC.get(kind, OFF_TOPIC["video"])
    return (f'For an unrelated request, reply with exactly this sentence (translated into '
            f'{config.SUMMARY_LANGUAGE} if that is another language): "{sentence}"')


def answer(parent_id: int, question: str, user_id: int, backend: str | None, model: str | None) -> AskResult:
    """Answers one question about a delivered summary.

    Args:
        parent_id: The summary request it is about.
        question: The reader's question (at most ASK_MAX_QUESTION characters, checked by the bot).
        user_id: Who asks (only their earlier questions are sent along).
        backend: The asker's backend; None = the default.
        model: The asker's model; None = the backend's default.

    Raises:
        pipeline.PipelineError: The summary is too old, or the AI failed (message for the user).
        proc.ProcCancelled: The job was cancelled.
    """
    text = prompt(parent_id, question, user_id)
    system = summarize.FOLLOWUP_SYSTEM + "\n" + off_topic_rule(db.get_delivered(parent_id)["kind"])
    try:
        got, answered_by = summarize.ask(backend, model, system, text, summarize.ANSWER_SCHEMA)
    except summarize.SummaryError as e:
        raise pipeline.PipelineError(str(e), detail=e.detail)
    llm = summarize.llm_label(backend, answered_by or model or "")
    return AskResult(question=question, answer=(got.get("answer") or "").strip(), parent_id=parent_id, llm=llm)
