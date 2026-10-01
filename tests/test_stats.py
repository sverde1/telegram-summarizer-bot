"""Measured speeds and remembered values."""
from summarizer import stats


def test_first_value_is_taken_then_moving_average():
    assert stats.get("k", 9.0) == 9.0
    stats.record("k", 10.0)
    assert stats.get("k", 0) == 10.0
    stats.record("k", 20.0)
    assert stats.get("k", 0) == 13.0  # 0.7 * 10 + 0.3 * 20


def test_remember_and_recall():
    assert stats.recall("model:codex") == ""
    stats.remember("model:codex", "gpt-x")
    assert stats.recall("model:codex") == "gpt-x"
