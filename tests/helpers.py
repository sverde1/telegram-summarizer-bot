"""Shared fakes for pipeline and summarize tests."""
from summarizer import summarize

SUMMARY = {"title": "T", "is_clickbait": False, "clickbait_answer": "", "summary": "A summary."}


def meta(**overrides) -> dict:
    """Probe metadata as media.probe returns it, for a 10-minute captioned YouTube video."""
    return {"id": "abcdefghijk", "title": "Test video", "description": "", "uploader": "U", "duration": 600,
            "upload_date": "20260101", "language": "en", "thumbnail": "", "subtitles": {"en": []},
            "auto_captions": [], "is_carousel": False, "music": "", **overrides}


class FakeConversation(summarize.Conversation):
    """A scripted LLM: answers each turn with the next prepared answer and records what it was sent.

    Attributes:
        answers: Answers for turn 1, turn 2, … (dicts in the turn's schema).
        sent: (images, schema keys) per turn.
        closed: Whether close() was called.
    """

    def __init__(self, answers: list[dict], model: str = "fake-model"):
        """Prepares the scripted answers and the model name it reports."""
        self.answers = list(answers)
        self.sent: list[tuple[list, set]] = []
        self.closed = False
        self._model = model

    def _send(self, system, text, images, schema, first):
        """Returns the next scripted answer and reports the model, as a real backend would."""
        self.sent.append((list(images), set(schema["properties"])))
        self.model = self._model
        return self.answers.pop(0)

    def close(self):
        """Records that the pipeline cleaned up."""
        self.closed = True


class BookAI(summarize.Conversation):
    """A fake LLM for book summaries: answers by schema, using the chapter indices it was sent.

    Every instance is one conversation; the class keeps a log of all of them (`calls`) to check that each
    question got a fresh, closed conversation.
    """

    calls: list["BookAI"] = []

    def __init__(self, model: str = "fake-model"):
        """Starts an unused conversation."""
        self.text, self.schema, self.closed, self._model = "", None, False, model
        BookAI.calls.append(self)

    def _send(self, system, text, images, schema, first):
        """Answers in the requested schema: "S<i>" per chapter, a merged text, or a book summary."""
        import re
        assert first and not images  # books are single-turn, text-only
        self.text, self.schema, self.model = text, schema, self._model
        if "chapters" in schema["properties"]:
            return {"chapters": [{"index": int(i), "summary": f"S{i}"} for i in re.findall(r'index="(\d+)"', text)]}
        if "title" in schema["properties"]:
            return {"title": "Book T", "author": "Ana", "summary": "Whole book."}
        return {"summary": "merged"}

    def close(self):
        """Records that the conversation was closed."""
        self.closed = True
