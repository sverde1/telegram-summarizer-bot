"""Messages always fit Telegram, stay valid HTML, and model output can't turn into a message flood."""
import html
import re

import pytest

from summarizer import pipeline
from tgbot import render


def _result(title="T", answer="", summary="S", clickbait=False) -> pipeline.Result:
    """A summary result with the given fields."""
    return pipeline.Result("youtube", "id", "https://youtu.be/id", {"title": "T"}, "", "captions", "en",
                           {"title": title, "is_clickbait": clickbait, "clickbait_answer": answer, "summary": summary})


def _valid(chunk: str) -> bool:
    """Whether a message is balanced HTML with only complete entities (what Telegram's parser needs)."""
    without_tags = re.sub(r"</?(b|i)>", "", chunk)
    entities_ok = all(m in ("&amp;", "&lt;", "&gt;", "&quot;", "&#x27;")
                      for m in re.findall(r"&[^;\\s]*;?", without_tags))
    return entities_ok and chunk.count("<b>") == chunk.count("</b>") and chunk.count("<i>") == chunk.count("</i>")


@pytest.mark.parametrize("result", [
    _result(title="x" * 4200),
    _result(summary="• point\n" * 2000),
    _result(answer="&" * 1000, clickbait=True),
    _result(summary=("<&>" * 2000)),
    _result(summary="a" * 10_000),
])
def test_every_message_fits_and_is_valid(result):
    chunks = render.render(result)
    assert all(len(c) <= render.TG_LIMIT for c in chunks)
    assert all(_valid(c) for c in chunks)


def test_fields_are_capped():
    chunks = render.render(_result(title="t" * 5000, summary="s" * 50_000))
    title_line = html.unescape(chunks[0].split("\n")[1])
    assert len(title_line) == render.FIELD_LIMITS["title"] and title_line.endswith("…")
    assert len(chunks) <= 3  # 4000 summary characters at most, not a flood


def test_short_result_is_one_message_with_footer():
    chunks = render.render(_result())
    assert len(chunks) == 1 and chunks[0].startswith("<b>Title:</b>") and chunks[0].rstrip().endswith("</i>")
