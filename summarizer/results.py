"""What every finished job hands back to the bot: the common part of video, document and voice results.

The bot treats all results alike for caching, privacy pacing and the wait estimate, and asks the result itself
for the rest (its status line, its work time, its model). Imports nothing from the bot.
"""
from dataclasses import dataclass


@dataclass(kw_only=True)
class JobResult:
    """The fields and questions the bot has for any result.

    Keyword-only, so subclasses can keep their own positional fields in front.

    Attributes:
        cached: Whether the result came from the database without new work.
        hold: For a first-time requester whose answer reused earlier work: (status text, seconds) still owed
            before delivery, so the wait matches a fresh run. The bot waits them out off the worker.
        replay_steps: For a requester who mustn't learn that earlier work was reused: the (step, seconds) a
            fresh run shows. On a cached result the bot plays them back before delivering. None for everyone
            else.
        replay_total: Seconds those steps add up to.
    """
    cached: bool = False
    hold: list[tuple[str, float]] | None = None
    replay_steps: list[tuple[str, float]] | None = None
    replay_total: float = 0.0

    def head(self) -> str:
        """First line of the status message while the result is replayed (what it is about)."""
        return ""

    def work_seconds(self) -> float | None:
        """The worker's time for a fresh result, to teach the queue's wait estimate; None to not count it."""
        return None

    def llm_label(self) -> str:
        """The model that wrote the result, e.g. "Codex · gpt-6-sol", or "" when not known."""
        return ""
