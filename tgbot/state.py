"""The bot's in-memory state: the job queue and every job not yet finished, plus the job type itself.

Module attributes, always used as `state.x` (never imported by name), so a rebinding such as
`state.queue = asyncio.Queue()` or reset() is seen everywhere.
"""
import asyncio
import collections
import enum
import threading
from dataclasses import dataclass, field


class JobKind(enum.Enum):
    """What a job works on; each kind has its runner (runners.RUNNERS) and the Job fields it needs."""
    VIDEO = "video"  # a YouTube/TikTok link (url)
    MEDIA = "media"  # a sent recording: voice message, audio or video file, or a shared one (upload_id)
    DOCUMENT = "document"  # a book or document (upload_id, book_mode, chapter)
    VOICE = "voice"  # 🔊: reading a summary aloud (voice_of)
    ASK = "ask"  # 💬: a follow-up question about a summary (ask_of, question)


# Compared and hashed by identity: jobs go in sets (state.running), and two jobs are never the same job.
@dataclass(eq=False)
class Job:
    """One queued request: what to process, where to reply, and on whose behalf.

    Attributes:
        url: The link the user sent (a file's label for recordings and documents).
        job_kind: What it is (see JobKind); checked against the fields it needs.
        chat_id: Chat to reply in.
        status_id: Message id of the status message that gets edited while the job runs.
        use_cache: False for /again (re-summarize, ignoring the cached summary).
        transcript_only: True for /transcript (send the transcript, no summary).
        queued_at: time.monotonic() when the job was queued, to report time spent waiting.
        user_id: Telegram user id of the requester.
        request_id: Row id in the `requests` table, updated when the job finishes.
        backend: The user's chosen LLM backend; None = the default.
        model: The user's chosen model of that backend; None = the default.
    """

    url: str
    chat_id: int
    status_id: int
    use_cache: bool = True
    transcript_only: bool = False
    queued_at: float = 0.0
    user_id: int = 0
    request_id: int = 0
    backend: str | None = None  # the user's chosen LLM backend/model; None = default
    model: str | None = None
    cancel_reason: str | None = None  # set when the job is cancelled; the text shown to the user
    waiting_since: float | None = None  # when it was first set aside for lack of memory (monotonic)
    memory_needed: int = 0  # bytes its transcription needs (shown while it waits)
    audio_seconds: float = 0  # its audio length, to re-estimate the memory need on each re-check
    upload_id: int = 0  # an uploaded document (url is then its file name); 0 for links
    book_mode: str = ""  # whole | short | each | pick
    chapter: int | None = None  # for "pick": the chosen chapter (None: show the chapter list)
    ocr_ok: bool = False  # the user confirmed OCR of this scanned document
    voice_of: int = 0  # 🔊: the request whose summary to read aloud; 0 for everything else
    ask_of: int = 0  # 💬: the summary request a question is about; 0 for everything else
    question: str = ""  # 💬: the question
    reply_to: int = 0  # 💬: the question's message, which the answer replies to
    # Set to stop this job's programs (proc.job_cancel while it runs); jobs run side by side, so each has its own.
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False, compare=False)
    started_at: float = 0.0  # when the worker started it (monotonic); the status's elapsed time counts from it
    job_kind: JobKind = JobKind.VIDEO

    def __post_init__(self) -> None:
        """Rejects a kind without the fields it needs (or with another kind's), so no job is run as the wrong
        kind.

        Raises:
            ValueError: The fields don't fit the kind.
        """
        uploads = self.job_kind in (JobKind.MEDIA, JobKind.DOCUMENT)
        ask = self.job_kind is JobKind.ASK
        if (bool(self.upload_id) != uploads or bool(self.voice_of) != (self.job_kind is JobKind.VOICE)
                or bool(self.ask_of) != ask or bool(self.question) != ask):
            raise ValueError(f"a {self.job_kind.value} job with upload_id={self.upload_id}, "
                             f"voice_of={self.voice_of}, ask_of={self.ask_of}")


# The queue, drained by config.WORKERS workers side by side; the heavy CPU steps take turns (summarizer/cpu.py).
queue: asyncio.Queue[Job] = asyncio.Queue()
# Every job not yet finished, by request id, so a user's jobs can be found and cancelled; `running` are the
# ones a worker is on now (each stopped through its own Job.cancel_event).
jobs: dict[int, Job] = {}
running: set[Job] = set()
# Jobs per user, queued or running (see config.MAX_QUEUED_PER_USER). Decremented whenever a job ends,
# however it ends, so a user is never locked out by a job that's gone.
user_jobs: collections.Counter = collections.Counter()
# Jobs set aside because their transcription doesn't fit in memory right now; retried between other jobs.
waiting_for_memory: list[Job] = []
delayed: dict[int, asyncio.Task] = {}  # request id -> paced delivery task (see jobs.deliver_later)
pending_replied: dict[int, float] = {}  # user id -> when they last got the "still waiting for approval" reply
admin_error_noticed: dict[str, float] = {}  # error type -> when the admins were last told
block_noticed: dict[str, float] = {}  # platform -> when the admins were last told
# User id -> (summary request, when) after a 💬 tap: their next plain text is the question even if they closed
# the reply bar (only for ASK_WINDOW seconds, and never a message with a link).
asking: dict[int, tuple[int, float]] = {}
md_sent: dict[int, float] = {}  # request id -> when its 📄 file was last sent (a tap flood mustn't upload a stream)


def reset() -> None:
    """Empties all state (tests: each gets a fresh event loop, and an asyncio.Queue binds to the first one)."""
    global queue, jobs, running, user_jobs, waiting_for_memory, delayed, pending_replied, admin_error_noticed
    global block_noticed, md_sent, asking
    queue, jobs, running, user_jobs = asyncio.Queue(), {}, set(), collections.Counter()
    waiting_for_memory, delayed, pending_replied, admin_error_noticed, block_noticed = [], {}, {}, {}, {}
    md_sent, asking = {}, {}
