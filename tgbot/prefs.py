"""A user's effective preferences, read when a job is queued."""

from summarizer import config, db, summarize


def current_llm(uid: int) -> tuple[str, str, bool]:
    """Resolves which backend and model a user's summaries use.

    Args:
        uid: Telegram user id.

    Returns:
        (backend, model, is_default): the effective backend and model, and whether the user is on the
        defaults (made no choice, or their chosen backend is no longer installed).
    """
    backend, model = db.get_user_llm(uid)
    if backend not in summarize.available_backends():
        backend, model = None, None  # their choice was uninstalled: fall back
    b = backend or config.LLM_BACKEND
    return b, model or summarize.default_model(b), not backend and not model
