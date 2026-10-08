"""Several links in one message: one job for all of them.

The links' transcripts are made side by side (pipeline.run, transcript only: one lookup per link, cached
transcripts reused, Whisper in its CPU turn), duplicates of the same video dropped, then each link is
summarized as if it had been sent alone (reusing what's stored, so no second lookup). The bot delivers each
result with its own request (summary messages, buttons), exactly like single sends. Blocking: runs in a thread.

Privacy: each link runs with its own request id and hide_cache_from, so cached work is paced as usual; the time
owed is added up and waited out once, off the worker, before anything is delivered (GroupResult.hold).
"""
import logging
from dataclasses import dataclass, field

from . import db, memory, pipeline, proc
from .results import JobResult
from .urls import UnsupportedURL

log = logging.getLogger(__name__)

PARALLEL = 4  # links worked on at once within one message
FAILED = "Couldn't summarize this one."


@dataclass(kw_only=True)
class GroupResult(JobResult):
    """The outcome of a several-link message.

    Attributes:
        items: (the link's request id, its result), in the message's order.
        failures: (the link's request id, the link, the user message) for links that failed.
        count: How many links the message had.
    """
    items: list[tuple[int, pipeline.Result]] = field(default_factory=list)
    failures: list[tuple[int, str, str]] = field(default_factory=list)
    count: int = 0

    def head(self) -> str:
        """The status line while the owed time is waited out."""
        return f"🔗 {self.count} links"

    def work_seconds(self) -> float | None:
        """The links' work, added up (none: nothing fresh)."""
        return sum(r.work_seconds() or 0 for _, r in self.items) or None

    def llm_label(self) -> str:
        """The model of the first summary."""
        return next((r.llm_label() for _, r in self.items if r.llm_label()), "")


def _quiet(*args, **kwargs) -> None:
    """A status callback that shows nothing: the group has one status line for all its links."""
    proc.check_cancelled()  # still a cancel checkpoint, like the real status


def _parallel(fn, n: int) -> list:
    """fn(0..n-1) side by side in the job's pool; each result, or the exception it raised.

    Raises:
        Blocked, ProcCancelled, NeedsMemory: From any of them: they stop the whole group (a platform ban is
            reported, a cancel stops everything, a transcription that doesn't fit parks the group).
    """
    with proc.pool(min(PARALLEL, n)) as pool:
        futures = [pool.submit(fn, i) for i in range(n)]
        out = [f.exception() or f.result() for f in futures]
    for e in out:
        if isinstance(e, (pipeline.Blocked, proc.ProcCancelled, memory.NeedsMemory)):
            raise e
    return out


def _message(e: BaseException) -> str:
    """What a user sees for a link that failed (the fixed texts; anything unexpected is logged)."""
    if isinstance(e, (pipeline.PipelineError, UnsupportedURL)):
        return str(e)
    log.error("a link of a group failed", exc_info=e)
    return FAILED


def _owed(r: pipeline.Result) -> float:
    """Seconds a first-time requester is still owed for this result (see pipeline.plan_replay / hold)."""
    held = sum(sec for _, sec in r.hold or [])
    return held or (r.replay_total if r.cached and r.replay_steps else 0.0)


def run(urls: list[str], part_ids: list[int], progress, *, use_cache: bool, transcript_only: bool,
        backend: str | None, model: str | None, hide_cache_from: int | None,
        again_limit_user: int | None) -> GroupResult:
    """Works on the links of one message and returns their results.

    Args:
        urls: The links, in the message's order.
        part_ids: Each link's request row.
        progress: The job's status callback (text, eta).
        use_cache: False for /again (new summaries; transcripts are still reused).
        transcript_only: /transcript: the transcripts are the results.
        backend: The user's backend; None = the default.
        model: The user's model; None = the default.
        hide_cache_from: A requester who mustn't learn what others sent (pacing), or None for admins.
        again_limit_user: Whose /again cooldown applies, or None for admins.

    Raises:
        pipeline.Blocked, proc.ProcCancelled, memory.NeedsMemory: See _parallel.
    """
    n = len(urls)
    progress(f"🔗 {n} links · 🗣 getting the transcripts…", None)
    firsts = _parallel(lambda i: pipeline.run(urls[i], _quiet, transcript_only=True, request_id=part_ids[i],
                                              backend=backend, model=model, hide_cache_from=hide_cache_from), n)
    result = GroupResult(count=n)
    todo, seen = [], set()
    for i, r in enumerate(firsts):
        if isinstance(r, BaseException):
            result.failures.append((part_ids[i], urls[i], _message(r)))
        elif (r.platform, r.video_id) in seen:  # the same video twice: one summary
            db.update_request(part_ids[i], status="done", cached=1, error="the same video as another link")
        else:
            seen.add((r.platform, r.video_id))
            todo.append((i, r))
    if transcript_only:
        finals = [r for _, r in todo]
    else:
        progress(f"🔗 {n} links · 🧠 summarizing…", None)
        finals = _parallel(lambda k: pipeline.run(
            urls[todo[k][0]], _quiet, use_cache=use_cache, request_id=part_ids[todo[k][0]], backend=backend,
            model=model, hide_cache_from=hide_cache_from, again_limit_user=again_limit_user), len(todo))
    for (i, _), r in zip(todo, finals):
        if isinstance(r, BaseException):
            result.failures.append((part_ids[i], urls[i], _message(r)))
        else:
            result.items.append((part_ids[i], r))
    owed = sum(_owed(r) for _, r in result.items)
    for _, r in result.items:
        r.hold = None  # waited out once for the whole message
    if owed:
        result.hold = [(f"🔗 {n} links · 🧠 summarizing…", owed)]
    result.cached = bool(result.items) and all(r.cached for _, r in result.items)
    return result
