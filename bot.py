"""Telegram bot: send a link, a file or a book, get back a summary. The entry point (`python bot.py`).

The bot itself lives in the tgbot package; this file sets up logging, registers the handlers and runs it.
"""
import logging

import tgbot  # noqa: F401  (sets PTB_TIMEDELTA before telegram is imported)
from telegram import Update
from telegram.ext import (Application, CallbackQueryHandler, ChatMemberHandler, CommandHandler, MessageHandler,
                          filters)

from summarizer import config
from tgbot import handlers, intake, lifecycle

logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s", level=logging.INFO)
# httpx logs every Telegram API call (each long poll, each status edit) at INFO; that drowns the bot's log.
logging.getLogger("httpx").setLevel(logging.WARNING)


def add_handlers(app: Application) -> None:
    """Registers the bot's update handlers and error handler on an application.

    Separate from main() so tests can build an application with exactly the bot's handlers.
    """
    app.add_error_handler(lifecycle.on_error)
    # New messages only: an edited message would re-run a command (and edited ones carry no
    # update.message, which the handlers use to reply).
    # Private chats only (see handlers._private).
    new = filters.UpdateType.MESSAGE & filters.ChatType.PRIVATE
    app.add_handler(ChatMemberHandler(handlers.on_my_chat_member, ChatMemberHandler.MY_CHAT_MEMBER))
    for name, handler in [("start", handlers.on_start), ("help", handlers.on_help),
                          ("again", intake.command(use_cache=False)),
                          ("transcript", intake.command(transcript_only=True)), ("users", handlers.on_users),
                          ("history", handlers.on_history), ("models", handlers.on_models),
                          ("limit", handlers.on_limit), ("ocrlang", handlers.on_ocrlang),
                          ("units", handlers.on_units)]:
        app.add_handler(CommandHandler(name, handler, filters=new))
    # Order matters: the first matching handler wins, so the picker's `llm:` buttons must be registered
    # before the catch-all admin-button handler.
    app.add_handler(CallbackQueryHandler(handlers.on_llm_button, pattern=r"^llm:"))
    app.add_handler(CallbackQueryHandler(handlers.on_cancel_button, pattern=r"^cancel:"))
    app.add_handler(CallbackQueryHandler(intake.on_book_button, pattern=r"^book:"))
    app.add_handler(CallbackQueryHandler(handlers.on_voice_button, pattern=r"^voice:"))
    app.add_handler(CallbackQueryHandler(handlers.on_md_button, pattern=r"^md:"))
    app.add_handler(CallbackQueryHandler(handlers.on_units_button, pattern=r"^units:"))
    app.add_handler(CallbackQueryHandler(handlers.on_ocr_button, pattern=r"^ocr:"))
    app.add_handler(CallbackQueryHandler(handlers.on_ocr_admin_button, pattern=r"^ocradm:"))
    app.add_handler(CallbackQueryHandler(handlers.on_button, pattern=r"^(allow|block|remove):"))
    # Before on_message: a file sent with a caption is a file, not a message with a link (the attachment is
    # what gets summarized).
    app.add_handler(MessageHandler(new & (filters.VOICE | filters.AUDIO | filters.VIDEO | filters.VIDEO_NOTE),
                                   intake.on_media))
    app.add_handler(MessageHandler(new & filters.Document.ALL, intake.on_document))
    app.add_handler(MessageHandler(new & (filters.TEXT | filters.CAPTION) & ~filters.COMMAND
                                   & ~filters.Document.ALL, intake.on_message))


def main() -> None:
    """Builds the application, registers the handlers and runs long polling until stopped.

    Raises:
        SystemExit: If TELEGRAM_BOT_TOKEN isn't set.
    """
    if not config.TELEGRAM_BOT_TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN is not set (.env)")
    config.secure_files()
    app = (Application.builder().token(config.TELEGRAM_BOT_TOKEN).post_init(lifecycle.post_init)
           .post_stop(lifecycle.post_stop).build())
    add_handlers(app)
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
