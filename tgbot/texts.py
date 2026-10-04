"""Fixed user-facing messages shared by several modules."""


HELP = (
    "Send me a YouTube or TikTok link and I'll reply with the title, an answer to any clickbait, "
    "and a summary.\n\n"
    "/again <url> - ignore the cache and summarize again\n"
    "/transcript <url> - send the raw transcript as a file\n"
    "/history - your recent requests\n"
    "/models - show or choose the AI (Codex or Claude) and model\n"
    "/limit - how many requests and questions you have left today\n"
    "/units - measurements in metric or imperial, temperatures in °C or °F\n\n"
    "🎤 Send a voice message, audio or video file (up to 20 MB) to summarize it; with the caption /transcript "
    "you get the transcript instead.\n"
    "📄 You can also send a book or document (PDF, EPUB, DOCX or TXT, up to 20 MB), or a Google Drive or "
    "Dropbox link to one (shared as \"Anyone with the link\"): I'll summarize the whole thing or chapter by "
    "chapter.\n\n"
    "Under every summary: 💬 Ask to ask about it (or just reply to it), 📄 Download for a file to take to "
    "another chat."
)


ADMIN_HELP = ("\n\nAdmin:\n/users - list users; allow, remove, or unblock them\n"
              "/history - recent requests from all users (who sent what, cache hits)\n"
              "/limit - daily limits: /limit 50 (everyone), /limit <user id> 200 (one user), "
              "/limit <user id> default, /limit 0 (no limit); /limit ocr … for scanned documents\n"
              "/ocrlang - OCR languages: /ocrlang add slv, /ocrlang remove slv")


ACCESS_REMOVED = "⛔ Your access to this bot was removed, so this request was cancelled."


TOO_BIG = ("⚠️ This file is larger than 20 MB, the most Telegram lets bots download. Upload it to Google Drive "
           "or Dropbox, share it as \"Anyone with the link\" and send me the link (or send a smaller version, e.g. "
           "an EPUB).")


DOWNLOAD_FAILED = "⚠️ Couldn't download the file from Telegram. Please send it again."


MEDIA_TOO_BIG = ("⚠️ This file is larger than 20 MB, the most Telegram lets bots download. Upload it to Google "
                 "Drive or Dropbox, share it as \"Anyone with the link\" and send me the link.")


UNREADABLE_MEDIA = "⚠️ I couldn't read this file as audio or video."


CANCELLED = "✖️ Cancelled. You can send the link again any time."


TOO_OLD = "This summary is too old for that; ask for it again."
VOICE_TOO_OLD = "This summary is too old to read aloud; ask for it again."


INTERNAL_ERROR = "⚠️ Something went wrong on the bot's side."


INTERNAL_ERROR_NOTIFIED = INTERNAL_ERROR + " The admin has been notified."


STOPPED = "⏹ The bot was stopped before your summary was ready. Please send the link again later."
