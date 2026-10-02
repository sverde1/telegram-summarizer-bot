# AGENTS.md

Guidance for AI coding agents working on this repository. User-facing documentation is in
`README.md`.

## What this is

A private Telegram bot that summarizes YouTube and TikTok videos. Python 3, python-telegram-bot
(async, long polling), yt-dlp / gallery-dl for media, faster-whisper for speech-to-text, and an LLM
reached through the Codex CLI, the Claude Code CLI, the Anthropic API or the OpenAI API. It runs as a systemd user
service on a single machine with no GPU.

## Layout

| Path | Role |
|---|---|
| `bot.py` | Telegram handlers, job queue/worker, status message + ETA, rendering, `/models`, `/users`, `/history`, update notifications, uploads and their buttons (`on_document`, `on_book_button`). |
| `access.py` | Who may use the bot (admins from `.env`, others from the `users` table). |
| `summarizer/pipeline.py` | URL in, `Result` out: cache lookup, transcript, frames, LLM turns, timings. Blocking; runs in a worker thread. |
| `summarizer/summarize.py` | Prompts, JSON schemas, the three LLM backends as `Conversation` classes, model lists. |
| `summarizer/media.py` | yt-dlp / gallery-dl calls: probe, captions, audio, video, thumbnail, carousel slides. |
| `summarizer/transcribe.py` | faster-whisper (model load + self-test, ffmpeg decode, speed stats). |
| `summarizer/frames.py` | Frame grabbing at given moments, sweeps, near-duplicate removal. |
| `summarizer/db.py` | SQLite schema, migrations (in `init()`), all queries. |
| `summarizer/urls.py` | URL classification and normalisation. |
| `summarizer/stats.py` | Measured speeds and small remembered values (`data/stats.json`). |
| `summarizer/updates.py` | Checks for newer Codex / Claude Code releases. |
| `summarizer/config.py` | Settings from `.env`; puts the venv's `bin` on `PATH`; `clean_env()` for child processes; `umask 077`. |
| `summarizer/proc.py` | The only way to run external programs: own process group, killed on timeout or cancel (`current_job_cancel`), secrets stripped from the environment. |
| `summarizer/memory.py` | Whisper memory estimate and the RAM checks behind the memory guard. |
| `summarizer/sandbox.py` | bwrap command for code that touches untrusted files: no network, no repo, no environment. |
| `summarizer/docparse.py` | Reads PDF/EPUB/DOCX/TXT into pages + chapters. Runs **inside** the sandbox; imports no bot module. |
| `summarizer/books.py` | Book summaries: chapters batched per call, long chapters in parts, whole book (map-reduce), cached in `doc_summaries`. |
| `summarizer/ocr.py` | OCR of scanned PDFs: rendering, Tesseract/RapidOCR in the sandbox, parallel batches, language check, estimates. |
| `summarizer/rapid_ocr.py` | RapidOCR entry point **inside** the sandbox (offline: models by path only). |
| `summarizer/links.py` | Google Drive / Dropbox share links: parsed into a bot-built download URL, downloaded via `fetch`. |
| `summarizer/fetch.py` | The only way to download a file by URL: HTTPS, allowed hosts checked on every redirect, size cap, optional SHA-256. |
| `summarizer/documents.py` | Bot side of docparse: runs it sandboxed, maps errors to user messages, stores pages by SHA-256. |
| `tests/` | pytest suite; `conftest.py` isolates tests from the real bot (see above), `helpers.py` has the fake LLM. |
| `deploy/` | systemd units and the weekly extractor-upgrade script. |
| `data/` | Runtime state, git-ignored: database, the bot's Codex login, temp job dirs. |

## Running and checking changes

```bash
.venv/bin/pip install -r requirements-dev.txt          # once: pytest, pytest-xdist, pytest-asyncio
.venv/bin/pytest                                       # the suite, in parallel on all cores
.venv/bin/pytest -n0 tests/test_bot.py -k progress     # one test, no parallelism (debugging, --pdb)
.venv/bin/pytest -m slow                               # real ffmpeg/Whisper work
.venv/bin/pytest -m network -n0                        # real downloads/LLM calls (owner's limits!)
systemctl --user restart telegram-summarizer-bot       # apply changes to the live bot
journalctl --user -u telegram-summarizer-bot -n 50 -o cat
```

Every change adds or updates tests, and the whole suite must pass before committing.

How the tests are isolated (`tests/conftest.py`):
- The real `.env` is never loaded and every test gets its own database and stats file in a temp dir;
  tests never touch `data/`.
- Network access and external programs are blocked (only ffmpeg and the test's own Python may run), so
  no test can reach YouTube, Telegram or an LLM unless it's marked `network`.
- Telegram is faked behind a real python-telegram-bot `Application` with the bot's own handlers
  (`telegram` and `app` fixtures, `msg_update` / `callback_update` / `send` helpers): filters, handler
  order and Telegram errors behave like in production.
- LLMs are faked with `tests/helpers.FakeConversation`; media with monkeypatched `summarizer.media`
  functions; `tiny_video` / `blue_video` are generated locally with ffmpeg.
- `network` tests use the owner's ChatGPT / Claude limits: keep them few and small.

## Rules that must hold

**Security (video content is untrusted and can contain prompt injection):**
- The LLM must never get tools, file access or network beyond its own API. Codex runs only inside the
  `bwrap` sandbox from `summarize._bwrap` (no `/home`, `.env` or repo mounted); Claude Code runs with
  `--tools ""` and `--strict-mcp-config`; the two API paths send no tools. Don't weaken these.
- Prompts treat titles, descriptions, transcripts and image text as data, never as instructions.
- Codex's tool features are switched off (`summarize.CODEX_DISABLED_FEATURES`, filtered against the
  installed Codex, which rejects unknown names). New Codex tool features must be added to that list.
- Book calls go through `summarize.ask`: a fresh conversation per call, always closed (never reuse one
  across chapters: APIs would resend everything, Claude Code would reuse a session id).
- Continue Codex conversations by the exact session id from the first turn, never `--last`; jobs
  from different users must not mix.
- Run external programs only through `proc.run` (never `subprocess` directly): it kills the whole
  process tree on timeout/cancel and strips the bot's secrets from the child's environment.
- Never swallow `proc.ProcCancelled` (a cancelled job must stop) or `media.Blocked` (a platform ban must
  be reported, not summarized around): code that tolerates failed downloads re-raises both.
- Uploaded files are parsed only inside `sandbox.command` (via `documents.parse`), never in the bot's
  process. The sandbox mounts `summarizer/` alone, never the repo (`.env` is there); code running inside
  (`docparse`) must not import `summarizer.config` or other bot modules. EPUB/DOCX members are read only
  through `docparse._read` (size and compression-ratio caps).
- OCR models are downloaded only by `ocr.add_language` (admins' `/ocrlang`): fixed URLs, HTTPS redirects
  only to the allowed hosts, size caps, SHA-256 for RapidOCR, a sandboxed test run for Tesseract models.
  The OCR sandbox is offline: nothing may download at OCR time.
- Document share links go only through `links.parse` (the download URL is rebuilt from the file id/path,
  never the user's URL) and `fetch.download` with the service's host allow-list.
- Links are accepted only through `urls.check` / `urls.classify` (host allow-list; TikTok short-link
  redirects validated hop by hop). Never fetch a user-supplied URL any other way.
- The bot answers only in private chats; every message/command handler is filtered to private chats and
  every button handler checks the chat type; the admins come from `.env`, buttons re-check access.
- Never commit `.env`, `data/`, tokens or logins.

**Errors:** users only see expected, fixed messages. Raise `PipelineError` / `SummaryError` with a
user message and put raw tool output, paths or exception text in `detail` (admins see it, users never).
Anything unexpected becomes the generic "something went wrong" message plus an admin notice.

**Abuse limits** (keep them when changing the queue): the OCR limit (`requests.ocr`, set when an OCR run
starts; cancelled runs count), the OCR page cap with admin approval (`ocr_holds`), the daily limit (global in the `settings` table,
per-user override in `users.daily_limit`, read from the database on each check), per-user and total queue
limits, the `/again` cooldown, the pending-request cap and reply throttling, the memory guard, and the
frame-grab cap. Admins are exempt from the daily and queue limits.

**Privacy:** who submitted which video is visible only to admins. Regular users may only learn that a
result was cached if they requested that video themselves (`db.user_saw_video`). Otherwise every reuse is
paced like a fresh run at half its time, at most 2 min: cached summaries and transcripts get a replay plan
(`pipeline.plan_replay`, played back by `bot._deliver_later`), a saved transcript reused for a new summary
gets a paced transcript step (`pipeline._pause`); the footer shows `Result.replay_steps`, so it always
matches the wait. Status lines must not mention the cache (`hide_cache`).

**Platform gotchas** (each was hit in practice):
- Run yt-dlp / gallery-dl from the venv (`config.YTDLP`), never a system binary.
- Probe TikTok via the `/video/` URL (yt-dlp rejects `/photo/`). Carousels show up as posts without
  video formats, or without any formats when there's no music (`--ignore-no-formats-error`).
- Whisper: always auto-detect the language and use multilingual models; forcing English on other
  languages produces fluent invented text.
- Audio is decoded with ffmpeg in `transcribe._decode`, not faster-whisper's PyAV path (new PyAV
  breaks it).
- Telegram: messages are HTML (`parse_mode=HTML`), so escape everything with `html.escape`; split at
  4096 characters; status edits are throttled and ordered by `Progress`.

**Data:**
- Schema changes go in `db.SCHEMA` plus a migration in `db.init()` for existing databases
  (`ALTER TABLE` when a column is missing). Never lose existing rows.
- Summaries are cached per video **and** backend + model (`summaries` table); transcripts per video.

## Conventions

- Match the existing style: small functions and type hints.
- Every function and method must have a Google-style docstring
  (https://google.github.io/styleguide/pyguide.html#383-functions-and-methods).
  Start with a one-line summary, then add `Args:`, `Returns:`, and `Raises:`
  sections when they are not obvious from the signature. This applies to
  private helpers and test helpers too — not just public APIs.
- Comment the *why*, not the *what*: any non-obvious decision, workaround,
  invariant, magic number, or ordering dependency gets a `#` comment explaining
  the reason (link the issue/ticket if one exists). Don't narrate code that
  reads plainly.
- User-facing text is short and plain, with one emoji per status line (🔎 🎧 🗣 🎞 🧠 ✅ ⚠️).
- New settings go in `summarizer/config.py`, `.env.example` and the README's configuration table.
- New commands need a handler in `main()`, an entry in the per-user command menus (`USER_COMMANDS` /
  `ADMIN_COMMANDS`), and a line in `HELP` and the README.
- Keep `README.md` in sync with behavior changes.
- Commit messages: imperative subject line, a short body explaining why when it isn't obvious.

## License

GPL-3.0-or-later (see `LICENSE`). Contributions are under the same license.
