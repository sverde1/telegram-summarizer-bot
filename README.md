# Telegram video summarizer

Private Telegram bot: send a YouTube or TikTok link, get back

```
Title:            the video title
Clickbait answer: one paragraph answering the title/thumbnail teaser (if it is clickbait)
Summary:          overview + key points, sized to fit a Telegram message
```


## Pipeline
1. Classify/normalise the URL (YouTube watch/shorts/youtu.be, TikTok video/photo/short links).
2. `yt-dlp` metadata probe. SQLite cache in `data/` keyed by (platform, id): a video is never fetched twice.
3. Transcript: YouTube captions -> audio + faster-whisper. TikTok: audio + whisper -> TikTok captions.
   Photo carousels: the slide images are the content.
4. One LLM conversation per video. Turn 1: metadata + thumbnail + timestamped transcript -> summary, plus
   `needs_frames` and the moments to look at. Turn 2 (only if it asked): the bot grabs frames at those
   moments (plus an even sweep for short or speechless videos), sends them in the same conversation, and
   the LLM revises the summary. Codex continues by the exact session id from turn 1 (never `--last`);
   Claude Code by a per-video UUID. Session files are deleted after each job.
5. Reply in Telegram (HTML, split at 4096 chars). Downloaded media is deleted after each job.

## LLM backends and sandboxing
Titles, descriptions, transcripts and on-screen text are untrusted, so the model gets no capabilities
beyond returning its JSON answer:

| `LLM_BACKEND` | Billing | Isolation |
|---|---|---|
| `codex` (default) | ChatGPT subscription | `codex exec` runs inside `bwrap`: read-only `/usr` + certs, the bot's own `CODEX_HOME`, and this job's images. No `/home`, no `.env`, no repo. Browser/computer-use/apps/web search disabled, Codex sandbox read-only, `--ephemeral`. |
| `claude-code` | Claude subscription | `claude -p --tools ""`: no tools at all, no MCP servers; images are sent inline. |
| `api` | Anthropic API key | Plain Messages call, no tools. |

The Codex backend uses its own login in `data/codex-home` (separate from `~/.codex`, so your
sessions and history stay invisible to it). Log in once:
```bash
CODEX_HOME=$PWD/data/codex-home codex login --device-auth
```

## Setup
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # fill in TELEGRAM_BOT_TOKEN (+ backend settings)
.venv/bin/python bot.py
```
With `ADMIN_USER_IDS` empty the bot replies with your user id; put it in `.env` and restart.

## Users
Admins (`ADMIN_USER_IDS`) approve everyone else from Telegram. When an unknown user starts the bot by
sending `/start`, the admins get an access request with **Allow / Deny** buttons; denied users are blocked and ignored
silently. `/users` lists allowed, pending and blocked users with Remove / Allow / Unblock buttons.
The list lives in `data/users.json`. All users share the admin's LLM subscription limits.
No system packages are needed: ffmpeg comes from `imageio-ffmpeg` (a system ffmpeg is preferred if
present) and deno (yt-dlp's YouTube JS runtime) from pip.

Long polling needs the bot to have no webhook set (`deleteWebhook`).

Run it permanently with the systemd user unit in `deploy/` (instructions inside).

## Commands
`/again <url>` re-summarize ignoring the cached summary · `/transcript <url>` raw transcript as .txt ·
`/users` (admins). The Telegram command menu is per user: strangers only see `/start`.

## GPU later
Set `WHISPER_DEVICE=cuda`, `WHISPER_COMPUTE_TYPE=float16`, `WHISPER_MODEL=medium`, install
`nvidia-cublas-cu12 nvidia-cudnn-cu12`, and uncomment `LD_LIBRARY_PATH` in the service unit. The bot
runs a 1-second test inference at model load and falls back to CPU if CUDA is broken.

## Maintenance
Platforms break extractors often. A weekly systemd user timer runs `deploy/update-extractors.sh`
(upgrades yt-dlp, its YouTube JS helpers, gallery-dl, deno); no bot restart needed. Install:
```bash
cp deploy/telegram-summarizer-update.{service,timer} ~/.config/systemd/user/
systemctl --user daemon-reload && systemctl --user enable --now telegram-summarizer-update.timer
journalctl --user -u telegram-summarizer-update   # what changed
```
For start-at-boot without logging in: `loginctl enable-linger $USER`.
