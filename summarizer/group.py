"""Several links in one message: one job for all of them.

The links' transcripts are made side by side (pipeline.run, transcript only: one lookup per link, cached
transcripts reused, Whisper in its CPU turn), duplicates of the same video dropped, then each link is
summarized as if it had been sent alone (reusing what's stored, so no second lookup). The bot delivers each
result with its own request (summary messages, buttons), exactly like single sends. Blocking: runs in a thread.

Privacy: each link runs with its own request id and hide_cache_from, so cached work is paced as usual; the time
owed is added up and waited out once, off the worker, before anything is delivered (GroupResult.hold).
"""
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field

from . import config, cpu, db, memory, pipeline, proc, summarize
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
    if not n:  # e.g. every link was a part of one series
        return []
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


# ---------- parts of one video ----------

# A part marker, the same kind in every part: "part 2" (in several languages), "2/4" (the same total), "(2)".
_WORD = re.compile(r"\b(?:part|pt|parte|teil|partie|del|deo|dio|deel|część|cześć|část|časť|часть)"
                   r"\.?\s*#?\s*(\d{1,2})\b", re.I)
_OF = re.compile(r"(?<![\d/])(\d{1,2})\s*/\s*(\d{1,2})(?![\d/])")
_PAREN = re.compile(r"\((\d{1,2})\)")
SERIES_MAX_FACTOR = 3  # a series may be this many times MAX_DURATION_MIN in all; longer ones stay separate


def part_numbers(texts: list[str]) -> list[int] | None:
    """The part numbers in a set of titles (or descriptions), when every one has the same kind of marker.

    "part N" in any of several languages, "N/M" with the same M (and N ≤ M), or "(N)". The numbers must be
    distinct: "1/2 cup" in two recipes isn't a series.

    Returns:
        The numbers in the given order, or None.
    """
    for rx in (_WORD, _OF, _PAREN):
        found = [rx.search(t or "") for t in texts]
        if not all(found):
            continue
        nums = [int(m.group(1)) for m in found]
        if rx is _OF and (len({m.group(2) for m in found}) != 1 or any(n > int(found[0].group(2)) for n in nums)):
            continue
        if len(set(nums)) == len(nums) and all(n >= 1 for n in nums):
            return nums
    return None


def _creator(r: pipeline.Result) -> str:
    """Who made a video, as an id (old cached metadata has only the uploader's name)."""
    return f"{r.platform}:{r.meta.get('uploader_id') or r.meta.get('uploader') or ''}"


def _plain_title(title: str) -> str:
    """A title without hashtags, case and spacing: parts reposted with the same caption compare equal."""
    return " ".join(re.sub(r"#\w+", "", title or "").lower().split())


def find_series(results: list[pipeline.Result], numbers: list[int | None] | None = None) -> list[list[int]]:
    """Which results are parts of one video: lists of indexes in part order (each with at least 2).

    Only one creator's videos can be parts of one video. Among them:
    - the ones whose titles (else descriptions) carry the same kind of part marker with distinct numbers, in
      that order (when only some have markers, those form the series and the rest stay separate);
    - else the ones with the same title (TikTok parts are often posted with one caption, the part number
      only on screen), ordered by the user's list numbers, then upload time, then the message's order.

    Args:
        results: The links' transcript results, in the message's order.
        numbers: Each link's number in the user's numbered list, if any.
    """
    numbers = numbers or [None] * len(results)
    by_creator: dict[str, list[int]] = {}
    for i, r in enumerate(results):
        by_creator.setdefault(_creator(r), []).append(i)
    series = []
    for creator, idx in by_creator.items():
        if len(idx) < 2 or creator.endswith(":"):
            continue
        for field_ in ("title", "description"):
            marked = [i for i in idx if _WORD.search(results[i].meta.get(field_) or "")]
            for subset in (idx, marked):
                if len(subset) < 2:
                    continue
                nums = part_numbers([results[i].meta.get(field_) or "" for i in subset])
                if nums:
                    series.append([i for _, i in sorted(zip(nums, subset))])
                    break
            else:
                continue
            break
        else:  # no markers: the same title
            by_title: dict[str, list[int]] = {}
            for i in idx:
                if title := _plain_title(results[i].meta.get("title", "")):
                    by_title.setdefault(title, []).append(i)
            for same in by_title.values():
                if len(same) >= 2:
                    series.append(sorted(same, key=lambda i: (numbers[i] if numbers[i] is not None else 10 ** 9,
                                                             results[i].meta.get("timestamp") or 0, i)))
    return series


def series_key(results: list[pipeline.Result]) -> str:
    """A series' id: the same parts give the same id in any order, so a resend is found in the cache."""
    ids = sorted(f"{r.platform}:{r.video_id}" for r in results)
    return hashlib.sha256("|".join(ids).encode()).hexdigest()[:32]


def _material(parts: list[pipeline.Result]) -> str:
    """The summary's material for a series: every part's metadata and transcript, in order."""
    blocks = []
    for n, r in enumerate(parts, 1):
        description = (r.meta.get("description") or "")[: 3000 if n == 1 else 1000]
        blocks.append(f"=== Part {n} of {len(parts)} ===\nTitle: {r.meta.get('title', '')}\n"
                      f"Duration: {r.meta.get('duration') or 0} s\nDescription:\n{description or '(none)'}\n"
                      f"<transcript>\n{r.transcript or '(no speech found)'}\n</transcript>")
    first = parts[0].meta
    return (f"This is one video posted in {len(parts)} parts (platform: {parts[0].platform}, uploader: "
            f"{first.get('uploader', '')}). Summarize the whole content as one video. There is no thumbnail: "
            "judge clickbait by the titles only. For title, give the video's title without the part number.\n\n"
            + "\n\n".join(blocks))


def _transcript(parts: list[pipeline.Result]) -> str:
    """The series' stored transcript: the parts' transcripts under part headings (💬 and 📄 read it)."""
    return "\n\n".join(f"=== Part {n}: {r.meta.get('title', '')} ===\n{r.transcript or '(no speech found)'}"
                        for n, r in enumerate(parts, 1))


def _make_series(parts: list[pipeline.Result], urls: list[str], *, use_cache: bool, backend: str | None,
                 model: str | None, hide: bool, started: float) -> pipeline.Result:
    """One video made of parts: stored as a pseudo-video ("series"), summarized once, cached per model."""
    key = series_key(parts)
    backend = backend or config.LLM_BACKEND
    model = model or summarize.default_model(backend)
    sources = sorted({r.transcript_source for r in parts})
    meta = {"title": "", "duration": sum(r.meta.get("duration") or 0 for r in parts), "parts": [
        {"url": u, "title": r.meta.get("title", ""), "duration": r.meta.get("duration") or 0}
        for u, r in zip(urls, parts)], "uploader": parts[0].meta.get("uploader", "")}
    saved = db.get_summary("series", key, backend, model) if use_cache and model else None
    cached = saved is not None
    if cached:
        summary = saved["result"]
    else:
        t = time.monotonic()
        try:
            answer, answered_by = summarize.ask(backend, model, summarize.SYSTEM, _material(parts),
                                                summarize.SCHEMA)
        except summarize.SummaryError as e:
            raise pipeline.PipelineError(str(e), detail=e.detail)
        took = time.monotonic() - t - cpu.waited(t)
        summary = {k: answer[k] for k in summarize.SCHEMA["required"]}
        total = time.monotonic() - started - cpu.waited(started)
        summary["_stats"] = {"steps": [("transcripts", max(0.0, total - took)), ("summary", took)], "total": total,
                             "llm": summarize.llm_label(backend, answered_by or model), "backend": backend,
                             "model": answered_by or model}
        db.save_summary("series", key, backend, model, summary, False)
    meta["title"] = summary.get("title") or parts[0].meta.get("title", "")
    db.start_video("series", key, "\n".join(urls))
    db.update_video("series", key, title=meta["title"][:300], meta=meta, transcript=_transcript(parts),
                    transcript_source=", ".join(sources), language=parts[0].language, status="done")
    result = pipeline.Result("series", key, "\n".join(urls), meta, _transcript(parts), ", ".join(sources),
                             parts[0].language, summary, cached=cached)
    if cached and hide:
        pipeline.plan_replay(result, False, backend)
    return result


def run(urls: list[str], part_ids: list[int], progress, *, use_cache: bool, transcript_only: bool,
        backend: str | None, model: str | None, hide_cache_from: int | None,
        again_limit_user: int | None, numbers: list[int | None] | None = None) -> GroupResult:
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
        numbers: Each link's number in the user's numbered list, if any (a hint for the parts' order).

    Raises:
        pipeline.Blocked, proc.ProcCancelled, memory.NeedsMemory: See _parallel.
    """
    n, started = len(urls), time.monotonic()
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
    series_items = []  # (message position, the first part's request id, the series result)
    if not transcript_only:
        for idx in find_series([r for _, r in todo], [numbers[i] for i, _ in todo] if numbers else None):
            parts = [todo[k] for k in idx]
            length = sum(r.meta.get("duration") or 0 for _, r in parts)
            if length > config.MAX_DURATION_MIN * 60 * SERIES_MAX_FACTOR:
                continue  # too long in all to summarize as one: the parts stay separate
            progress(f"🔗 {n} links · 🧠 summarizing the {len(parts)} parts as one video…", None)
            series = _make_series([r for _, r in parts], [urls[i] for i, _ in parts], use_cache=use_cache,
                                  backend=backend, model=model, hide=hide_cache_from is not None, started=started)
            first = parts[0][0]
            db.update_request(part_ids[first], platform="series", video_id=series.video_id)
            for i, _ in parts[1:]:  # the other parts are delivered with the first one
                db.update_request(part_ids[i], status="done", cached=int(series.cached))
            series_items.append((min(i for i, _ in parts), part_ids[first], series))
            owed_parts = sum(_owed(r) for _, r in parts)
            if owed_parts:  # cached transcripts a first-time requester is still owed
                series.hold = [("", owed_parts)]
            done = {i for i, _ in parts}
            todo = [(i, r) for i, r in todo if i not in done]
    if transcript_only:
        finals = [r for _, r in todo]
    else:
        progress(f"🔗 {n} links · 🧠 summarizing…", None)
        finals = _parallel(lambda k: pipeline.run(
            urls[todo[k][0]], _quiet, use_cache=use_cache, request_id=part_ids[todo[k][0]], backend=backend,
            model=model, hide_cache_from=hide_cache_from, again_limit_user=again_limit_user), len(todo))
    placed = []
    for (i, _), r in zip(todo, finals):
        if isinstance(r, BaseException):
            result.failures.append((part_ids[i], urls[i], _message(r)))
        else:
            placed.append((i, part_ids[i], r))
    placed += series_items
    result.items = [(pid, r) for _, pid, r in sorted(placed, key=lambda x: x[0])]
    owed = sum(_owed(r) for _, r in result.items)
    for _, r in result.items:
        r.hold = None  # waited out once for the whole message
    if owed:
        result.hold = [(f"🔗 {n} links · 🧠 summarizing…", owed)]
    result.cached = bool(result.items) and all(r.cached for _, r in result.items)
    return result
