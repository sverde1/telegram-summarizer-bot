"""Talking to Telegram for a job: the throttled status message and retried sends."""
import asyncio
import datetime as dt
import logging
import time

from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import Forbidden, RetryAfter, TelegramError
from telegram.ext import Application

from summarizer import proc
from tgbot import render, state

log = logging.getLogger("bot")  # one logger name for the whole bot, as in the journal


class Progress:
    """Shows pipeline stages in the status message, with an ETA countdown refreshed every 15 s.

    Called from the worker thread. Edits run in call order (asyncio.Lock is FIFO), so a later
    stage is never overwritten by an earlier one, and no stage is dropped.
    """

    # Seconds between countdown refreshes. Edits count against Telegram's per-chat rate limit
    # (about one message per second), so the countdown stays well below it.
    TICK = 15
    # A step expected to take at least this long gets a cancel button (and keeps it until the job ends): the
    # user may decide a long transcription or OCR isn't worth the wait. Quick jobs don't flash one.
    CANCEL_FROM = 60

    def __init__(self, app: Application, loop: asyncio.AbstractEventLoop, job: state.Job):
        """Starts the countdown ticker for one job's status message.

        Args:
            app: The running application (for the bot).
            loop: The bot's event loop; edits are scheduled onto it from the worker thread.
            job: The job whose status message to edit. Its started_at, when set, is where "elapsed" counts
                from, so a job continued off the worker (a hold or replay) doesn't restart at 0:00.
        """
        self.app, self.loop, self.job = app, loop, job
        self.lock = asyncio.Lock()
        self.text, self.eta, self.eta_at, self.shown, self.last_edit = "", None, 0.0, "", 0.0
        self.started, self.closed = job.started_at or time.monotonic(), False
        self.cancellable = False  # set once a long step showed the cancel button
        self.ticker = asyncio.run_coroutine_threadsafe(self._tick(), loop)

    def __call__(self, text: str, eta: float | None = None) -> None:
        """Shows a new stage. Called by the pipeline from the worker thread.

        Args:
            text: The full status text (all lines).
            eta: Estimated seconds until the summary is ready, or None if unknown.
        """
        self.text, self.eta, self.eta_at = text, eta, time.monotonic()
        # The pipeline runs in a worker thread; Telegram calls must run on the bot's event loop. The stage is
        # passed along: rendering the *current* text instead would skip a stage that's replaced by the next
        # one before its edit runs.
        asyncio.run_coroutine_threadsafe(self._edit((text, eta, self.eta_at)), self.loop)

    async def close(self) -> None:
        """Stop updating; waits for an in-flight edit so it can't overwrite the final message."""
        self.ticker.cancel()
        async with self.lock:
            self.closed = True

    def _render(self, stage: tuple | None = None) -> str:
        """Builds the status text: a stage plus elapsed time and the ETA countdown.

        Args:
            stage: (text, eta, eta_at) of a specific stage; None = the current one (countdown refresh).

        Returns:
            The text for the status message.
        """
        text, eta, eta_at = stage or (self.text, self.eta, self.eta_at)
        elapsed = time.monotonic() - self.started
        line = f"⏱ {int(elapsed) // 60}:{int(elapsed) % 60:02d} elapsed"
        if eta is not None:
            # Count down from when the ETA was given, not from job start: each stage brings a new ETA.
            left = eta - (time.monotonic() - eta_at)
            line += f" · ~{render.fmt_eta(left)} left" if left > 0 else " · taking longer than estimated…"
        return f"{text}\n\n{line}"

    async def _tick(self) -> None:
        """Refreshes the countdown every TICK seconds until cancelled."""
        while True:
            await asyncio.sleep(self.TICK)
            # Skip the refresh if a stage change just edited the message (TICK - 1 tolerates timer jitter):
            # that avoids a second, redundant edit right after it.
            if self.text and time.monotonic() - self.last_edit >= self.TICK - 1:
                await self._edit()

    async def _edit(self, stage: tuple | None = None) -> None:
        """Edits the status message, unless closed or unchanged.

        Args:
            stage: The stage to show (see _render); None = the current one.
        """
        # The lock serializes edits in call order, so a stage edit can't overtake a later one.
        async with self.lock:
            if self.closed:
                return
            text = self._render(stage)
            if text == self.shown:
                return
            try:
                eta = (stage or (self.text, self.eta))[1]
                self.cancellable = self.cancellable or (eta is not None and eta >= self.CANCEL_FROM)
                # Every edit must carry the button again: an edit without reply_markup removes it.
                markup = cancel_button(self.job.request_id) if self.cancellable else None
                await self.app.bot.edit_message_text(text, self.job.chat_id, self.job.status_id,
                                                     reply_markup=markup)
                self.shown, self.last_edit = text, time.monotonic()
            except RetryAfter as e:  # flood control: skip this edit, the next tick catches up
                log.warning("progress edit rate-limited for %ss", _retry_seconds(e))
            except TelegramError as e:
                # "message is not modified", the user blocked the bot, a network blip: a missed status edit
                # is harmless, and an exception here would kill the ticker task.
                log.debug("progress edit skipped: %s", e)


def _retry_seconds(e: RetryAfter) -> float:
    """Seconds Telegram asks us to wait. PTB returns an int today and a timedelta in a future version."""
    wait = e.retry_after
    return wait.total_seconds() if isinstance(wait, dt.timedelta) else float(wait)


class UserBlockedBot(Exception):
    """Telegram refuses to deliver to the user (they blocked the bot or deleted the chat)."""


def _check_not_cancelled(job: state.Job | None) -> None:
    """Raises ProcCancelled if the job was cancelled: nothing more may be sent for it."""
    if job is not None and job.cancel_reason:
        raise proc.ProcCancelled("cancelled")


async def send_with_retry(make_call, job: state.Job | None = None):
    """Runs one Telegram send, waiting out flood control once.

    Args:
        make_call: Zero-argument function returning the API coroutine (a coroutine can only be awaited once,
            so a retry needs a fresh one).
        job: The job the send belongs to. Checked before each attempt: a user removed while we waited out
            flood control must not get the result anyway.

    Raises:
        UserBlockedBot: Telegram says the user can't be reached.
        TelegramError: Any other Telegram failure, including a second flood-control refusal.
        proc.ProcCancelled: The job was cancelled before the send.
    """
    _check_not_cancelled(job)
    try:
        return await make_call()
    except RetryAfter as e:
        await asyncio.sleep(_retry_seconds(e))
        _check_not_cancelled(job)
        return await make_call()
    except Forbidden as e:
        raise UserBlockedBot(str(e))


def cancel_button(request_id: int) -> InlineKeyboardMarkup:
    """The ✖️ Cancel button for a job's status message (handled by handlers.on_cancel_button)."""
    return InlineKeyboardMarkup([[InlineKeyboardButton("✖️ Cancel", callback_data=f"cancel:{request_id}")]])
