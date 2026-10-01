# Telegram video summarizer

A private Telegram bot: send it a YouTube or TikTok link and it replies with

```
Title:
<video title, translated if it isn't in English>

Clickbait answer:
<one paragraph answering the title/thumbnail teaser, or "✅ Not clickbait">

Summary:
<overview, key points (including things only shown on screen), caveats>

⏱ 41 s total: lookup 2 s · Whisper 22 s · summary 17 s
📝 transcript: Whisper small (en) · 🧠 Codex (gpt-6-sol)
```

It reads the transcript (captions, or Whisper speech-to-text), looks at the thumbnail, and looks at
video frames or TikTok photo-carousel slides when the picture matters. Summaries are written by an LLM
running on your ChatGPT or Claude subscription, inside a sandbox.

## Features

- **YouTube** (videos, Shorts, `youtu.be`) and **TikTok** (videos, photo carousels, `vm.`/`vt.` links).
- **Transcripts:** YouTube captions first, otherwise local [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
  (CPU now, GPU-ready), language auto-detected.
- **Frames on demand:** the LLM reads the transcript and asks for frames when something is shown rather
  than said ("this book…"); they're sent in the same conversation and the summary is revised. Speechless
  videos get frames straight away; carousels send their slides.
- **Choice of AI per user** (`/models`): Codex (ChatGPT subscription) or Claude Code (Claude
  subscription), then a model. Default: Codex `gpt-6-sol`.
- **Live status with ETA** while it works; per-step timings under the result.
- **Multi-user with admin approval**: `/start` sends the admin an Allow / Deny request.
- **Cache**: a video is downloaded and transcribed once; summaries are kept per model.
- **Private by design**: users can't see what others submitted, and the LLM can't touch the machine.

## Requirements

- Linux with Python 3.11+ and `bwrap` (bubblewrap) for the Codex sandbox.
- A Telegram bot token from [@BotFather](https://t.me/BotFather).
- At least one LLM:
  - [Codex CLI](https://github.com/openai/codex) (`npm install -g @openai/codex`) and a ChatGPT plan, or
  - [Claude Code](https://code.claude.com) logged in with a Claude plan, or
  - an Anthropic API key.

No system ffmpeg is needed (`imageio-ffmpeg` provides one; a system ffmpeg is used if present), and
yt-dlp's YouTube JavaScript runtime (deno) comes from pip. No GPU needed.

## Installation

Step by step on Ubuntu/Debian; other Linux distributions work the same with their own package names.

### 1. System packages

```bash
sudo apt install git python3 python3-venv bubblewrap nodejs npm
```

`bubblewrap` (`bwrap`) sandboxes the Codex CLI; `nodejs`/`npm` are only needed to install Codex.

### 2. Get the code and install the Python dependencies

```bash
mkdir -p ~/scripts && cd ~/scripts
git clone https://github.com/sverde1/telegram-summarizer-bot.git
cd telegram-summarizer-bot
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

This installs yt-dlp, gallery-dl, faster-whisper, ffmpeg (via `imageio-ffmpeg`) and deno into the venv;
nothing else is needed system-wide. The Whisper model (~500 MB for `small`) downloads on first use.

### 3. Create the Telegram bot

1. In Telegram, open [@BotFather](https://t.me/BotFather), send `/newbot` and follow the prompts.
2. Copy the token it gives you.

### 4. Configure

```bash
cp .env.example .env
```

Edit `.env` and set `TELEGRAM_BOT_TOKEN=<token>`. Leave `ADMIN_USER_IDS` empty for now; the other
settings have working defaults (see [Configuration](#configuration-env)).

### 5. Connect an LLM

At least one of these; users can switch between the installed ones with `/models`.

**Codex (ChatGPT plan, the default):**
```bash
sudo npm install -g @openai/codex
CODEX_HOME=$PWD/data/codex-home codex login --device-auth
```
The bot gets its own Codex login in `data/codex-home`, separate from your personal `~/.codex`, so the
sandboxed process never sees your own Codex sessions. Follow the printed link and code to sign in.

**Claude Code (Claude plan):** install it from [code.claude.com](https://code.claude.com) and run
`claude` once to log in. The bot uses the `claude` command on `PATH`.

**Claude API:** set `ANTHROPIC_API_KEY` in `.env`.

### 6. First start and becoming admin

```bash
.venv/bin/python bot.py
```

1. Send your bot any message. With no admin configured it replies with your Telegram user id.
2. Stop the bot (Ctrl+C), set `ADMIN_USER_IDS=<your id>` in `.env`, and start it again.
3. Send it a YouTube or TikTok link.

The bot uses long polling. If the token was used with a webhook before (e.g. n8n), remove it first:
`curl "https://api.telegram.org/bot<token>/deleteWebhook"`.

### 7. Run it as a service

```bash
mkdir -p ~/.config/systemd/user
cp deploy/telegram-summarizer-bot.service deploy/telegram-summarizer-update.{service,timer} ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now telegram-summarizer-bot telegram-summarizer-update.timer
loginctl enable-linger $USER     # start at boot, keep running when logged out
```

The units expect the repo at `~/scripts/telegram-summarizer-bot`; edit the paths in them if it lives
elsewhere. Useful commands:

```bash
journalctl --user -u telegram-summarizer-bot -f        # live log
systemctl --user restart telegram-summarizer-bot       # after editing .env
```

### Updating

```bash
git pull
.venv/bin/pip install -r requirements.txt
systemctl --user restart telegram-summarizer-bot
```

The database migrates itself on start.

## Configuration (`.env`)

| Variable | Default | Meaning |
|---|---|---|
| `TELEGRAM_BOT_TOKEN` | – | Bot token (required). |
| `ADMIN_USER_IDS` | – | Comma-separated admin user ids. Empty = setup mode. |
| `LLM_BACKEND` | `codex` | Default provider: `codex`, `claude-code` or `api`. |
| `CODEX_MODEL` | `gpt-6-sol` | Default Codex model (empty = Codex's own default). |
| `CODEX_EFFORT` / `CLAUDE_EFFORT` | `medium` | Reasoning effort. |
| `CLAUDE_CODE_MODEL` / `CLAUDE_MODEL` | `claude-opus-5-5` | Default model for Claude Code / the API. |
| `ANTHROPIC_API_KEY` | – | Enables the `api` provider. |
| `SUMMARY_LANGUAGE` | `English` | Language of the summaries. |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda`. |
| `WHISPER_MODEL` | `small` | Multilingual models only (`small`, `medium`, `large-v3`). |
| `WHISPER_COMPUTE_TYPE` | `int8` | `float16` on a GPU. |
| `MAX_DURATION_MIN` | `180` | Longer videos are refused. |
| `MAX_FRAMES` / `MAX_SLIDES` | `16` / `35` | Images sent to the LLM. |
| `SHORT_VIDEO_SEC` | `180` | Up to this length, frames are also sampled every ~2 s. |

**GPU later:** set `WHISPER_DEVICE=cuda`, `WHISPER_COMPUTE_TYPE=float16`, `WHISPER_MODEL=medium`,
`pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`, and uncomment `LD_LIBRARY_PATH` in the service
unit. The bot test-runs the model at load and falls back to CPU if CUDA is broken.

## Commands

| Command | Who | |
|---|---|---|
| *(a link)* | users | Summarize it. |
| `/again <url>` | users | Summarize again with your current model, ignoring the cache. |
| `/transcript <url>` | users | The raw transcript as a `.txt` file. |
| `/history` | users | Your recent requests (admins: everyone's, with who sent them). |
| `/models` | users | Show or choose the AI: provider first, then model. |
| `/users` | admins | Users with Allow / Deny / Remove / Unblock buttons and their AI choice. |
| `/start` | strangers | Request access. |

The command menu adapts per user: strangers only see `/start`.

## How it works

1. **URL** → platform + id (tracking parameters stripped, TikTok short links resolved).
2. **Lookup** with yt-dlp. TikTok carousels are detected here (no video formats).
3. **Transcript:** YouTube captions → Whisper. TikTok: Whisper → TikTok's captions. Carousels: slides
   via gallery-dl instead. No or hardly any speech → frames are grabbed right away.
4. **LLM, turn 1:** metadata, thumbnail and timestamped transcript → summary, plus whether frames are
   needed and at which moments.
5. **LLM, turn 2 (only if asked):** frames from those moments, in the same conversation (Codex by its
   exact session id, Claude Code by a per-video UUID) → revised summary.
6. **Reply** in Telegram (HTML, split at 4096 characters). Downloaded media and LLM session files are
   deleted after each job.

Jobs run one at a time from a queue; the status message shows the queue position, each stage and an ETA
learned from this machine's measured speeds.

## Security and privacy

Video titles, descriptions, transcripts and on-screen text are untrusted: a video can contain prompt
injection. The model therefore gets no capability beyond returning its JSON answer:

| Provider | Isolation |
|---|---|
| Codex | `codex exec` inside `bwrap`: read-only `/usr` and certificates, the bot's own `CODEX_HOME`, and this job's images. No `/home`, `.env` or repo. Browser, computer use, apps and web search disabled; Codex's own sandbox read-only. |
| Claude Code | `claude -p --tools ""`: no tools, no MCP servers; images are sent inline. |
| Claude API | A plain Messages call with no tools. |

- Who submitted what is visible only to admins. A user is told a result came from the cache only if
  they requested that video themselves before (a cached reply is still faster, which can hint at it).
- All users share the admin's subscription limits.

## Data (`data/`, not in git)

`bot.sqlite3` holds four tables:

- `users`: id, name, username, status (admin / allowed / pending / blocked), chosen `backend` + `model`.
- `videos`: metadata and transcript, status `processing` → `done` / `failed`.
- `summaries`: one per video and model.
- `requests`: every link sent: who, what, kind, status, cache hit, timestamps.

Also there: `codex-home/` (the bot's Codex login), `stats.json` (measured speeds for ETAs).

## Maintenance

- **yt-dlp / gallery-dl** break when YouTube or TikTok change. The weekly timer runs
  `deploy/update-extractors.sh` to upgrade them; no restart needed.
- **Codex / Claude Code** are installed outside the venv. Every 12 h the bot checks npm for newer
  releases and messages the admins once per version with the update command. New models may need them.
- **New models:** Codex's list updates itself; the Claude Code list is in `summarizer/summarize.py`
  (`CLAUDE_CODE_MODELS`).


## License

Copyright (C) 2026 Sandi Verdev

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see [LICENSE](LICENSE)
for details.
