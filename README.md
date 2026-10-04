# Telegram video and book summarizer

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

It also summarizes **books and documents** (PDF, EPUB, DOCX, TXT, sent as a file or as a Google Drive /
Dropbox link): the whole book, or chapter by chapter. Scanned PDFs are read with OCR.

## Features

- **YouTube** (videos, Shorts, `youtu.be`) and **TikTok** (videos, photo carousels, `vm.`/`vt.` links).
- **Transcripts:** YouTube captions first, otherwise local [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
  (CPU now, GPU-ready), language auto-detected.
- **Frames on demand:** the LLM reads the transcript and asks for frames when something is shown rather
  than said ("this book…"); they're sent in the same conversation and the summary is revised. Speechless
  videos get frames straight away; carousels send their slides.
- **Choice of AI per user** (`/models`): Codex (ChatGPT subscription) or Claude Code (Claude
  subscription), or the Claude / OpenAI APIs with an API key, then a model. Default: Codex `gpt-6-sol`.
- **Live status with ETA** while it works; per-step timings under the result.
- **Multi-user with admin approval**: `/start` sends the admin an Allow / Deny request.
- **Books and documents**: PDF, EPUB, DOCX, TXT, uploaded (up to 20 MB) or as a Google Drive / Dropbox
  link (up to 100 MB); the whole book or chapter by chapter (short, one per message, or a picked chapter).
  Scanned PDFs are read with OCR (Tesseract or RapidOCR) after the user confirms the estimated time, with
  their own daily limit; admins add OCR languages with `/ocrlang`.
- **Voice messages, audio and video files** you send are summarized too (Whisper, frames for videos).
- **🔊 Listen**: any summary as a Telegram voice message (Kokoro text-to-speech, made on this machine),
  e.g. for the car.
- **Cache**: a video is downloaded and transcribed once, a document read (or OCRed) once; summaries are
  kept per model.
- **Private by design**: users can't see what others submitted, and the LLM can't touch the machine.

## Requirements

### Software

- Linux with Python 3.11+ and `bwrap` (bubblewrap) for the Codex and document sandboxes.
- poppler-utils (`pdftotext`, `pdftoppm`) for PDFs, and Tesseract for scanned documents: on the CPU it read
  a test book 5× faster than RapidOCR with the same text, and it checks a scan's script (Latin, Cyrillic, …)
  before OCR. With an NVIDIA GPU, RapidOCR (installed with the Python dependencies) is used instead.
- A Telegram bot token from [@BotFather](https://t.me/BotFather).
- At least one LLM:
  - [Codex CLI](https://github.com/openai/codex) (`npm install -g @openai/codex`) and a ChatGPT plan, or
  - [Claude Code](https://code.claude.com) logged in with a Claude plan, or
  - an Anthropic or OpenAI API key (billed per token).
- Text-to-speech for 🔊 voice messages: [Kokoro](https://github.com/thewh1teagle/kokoro-onnx)
  (`kokoro-onnx`, its v1.0 model and voices, and the system's espeak-ng: `sudo apt install espeak-ng-data`).
  **Why Kokoro:** compared with Piper on a real English summary (2¼ min of audio), Kokoro sounded much more
  natural, which matters for listening to whole summaries, e.g. in the car. It is slower on the CPU (61 s for
  that summary, 2.4× faster than real time, vs Piper's 7 s), so the voice message is made only when someone
  asks for it; with a GPU (`onnxruntime-gpu`) it takes seconds. Its int8 model was even slower here (0.6×
  real time) and isn't used.
- ffmpeg: optional. The bot ships one inside the venv (`imageio-ffmpeg`, a self-contained ffmpeg 7 build)
  and prefers a system ffmpeg (`sudo apt install ffmpeg`) when one is installed. Get the system one if you
  add a GPU: the bundled build can't use it. yt-dlp's YouTube JavaScript runtime (deno) also comes from pip.

### Hardware

| | Minimum | Recommended |
|---|---|---|
| CPU | 4 cores, x86-64 | 4+ modern cores |
| RAM | 4 GB free | 8 GB |
| Disk | 2 GB (venv ~1 GB, Whisper `small` ~0.5 GB) | 5 GB for larger Whisper models |
| GPU | none | NVIDIA with CUDA and 4+ GB VRAM, for Whisper |
| Network | broadband | broadband |

No GPU is needed: everything runs on CPU. A GPU only speeds up Whisper, the speech-to-text step for
videos without captions, and lets you use the more accurate `medium` / `large-v3` models
(see [GPU](#gpu)).

### Performance

Benchmarked on two desktops with the same test videos (a 10.6-minute captioned YouTube talk, a 1-minute
TikTok). LLM time isn't hardware-dependent and is listed separately.

| Step | Older desktop: 4th-gen Intel Core i5, 4 cores, 16 GB, no GPU | Newer desktop: 12th-gen Intel Core i9, 24 threads, 32 GB, NVIDIA RTX 30-series (12 GB) |
|---|---|---|
| YouTube lookup (yt-dlp) | 3.7 s | 3.6 s |
| YouTube captions | 1–17 s (YouTube throttles at times) | 4.9 s |
| Audio download (10.6-min video) | 4.7 s | 4.5 s |
| TikTok lookup + video download (1 min) | 3.9 s | 3.8 s |
| Frame grabbing (1-min video, 16 frames) | 3.0 s | 1.5 s |
| ffmpeg audio decode (10.6 min) | 1.7 s (ffmpeg 8.0) | 1.3 s (ffmpeg 7.0) |
| Whisper `small`, CPU (int8, 4 threads) | 169 s = 0.27× video length | 148 s = 0.23× |
| Whisper `small`, GPU (float16) | – | 22 s = 0.03× |
| Whisper `medium`, GPU (float16) | – | 39 s = 0.06× |
| Whisper `large-v3`, GPU (float16) | – | 88 s = 0.14× |

Whisper model load: a few seconds once downloaded (first download: `small` ~0.5 GB, `large-v3` ~3 GB);
the bot loads it once per start. The newer desktop ran under WSL2 on Windows.

Other steps, on either machine:

| Step | Time |
|---|---|
| LLM summary | 5–20 s per turn; a second turn when it asks for frames |
| Cached video | ~1 s for admins and repeat requests; others get half the original time (at most 2 min) |
| Memory | ~200 MB idle, up to ~1.5 GB with Whisper `small` loaded |

**What that means per video:** a captioned YouTube video takes 15–30 s anywhere. Speech-to-text dominates
everything else: a 30-minute video without captions takes ~8 minutes on the older CPU, ~1 minute with
the GPU and `small`, ~2 minutes with the GPU and the more accurate `medium`.

Whisper on CPU uses 4 threads by default, which is why the 24-thread CPU was barely faster there; set
`WHISPER_CPU_THREADS` to the number of physical cores on a bigger CPU. ffmpeg's share (audio conversion,
a few frame grabs) is seconds either way (older ffmpeg builds are slower: Ubuntu 24.04's 6.1 took 2.8 s for
the same decode); the bot runs it on the CPU, so a GPU and a system ffmpeg barely
change that part.

## Installation

Step by step on Ubuntu/Debian; other Linux distributions work the same with their own package names.

### 1. System packages

```bash
sudo apt install git python3 python3-venv bubblewrap nodejs npm poppler-utils tesseract-ocr espeak-ng-data
```

`bubblewrap` (`bwrap`) sandboxes the Codex CLI and the reading of uploaded files; `nodejs`/`npm` are only
needed to install Codex. `poppler-utils` reads PDFs; `tesseract-ocr` reads scanned ones; `espeak-ng-data`
(with its library) lets Kokoro read summaries aloud.
Further OCR languages are added from Telegram with `/ocrlang`, no system packages needed.

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
3. Turn off group chats, since the bot is for private chats only: in BotFather send `/mybots`, pick the bot,
   then **Bot Settings → Allow Groups? → Turn groups off**. (If it's added to a group anyway, it leaves at
   once and tells the admins who added it.)

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

**Claude API / OpenAI API:** set `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` in `.env`. These bill per token
on your API account, separately from ChatGPT or Claude plans.

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
| `LLM_BACKEND` | `codex` | Default provider: `codex`, `claude-code`, `api` (Claude) or `openai-api`. |
| `CODEX_MODEL` | – (`.env.example`: `gpt-6-sol`) | Default Codex model (empty = Codex's own default). |
| `CODEX_EFFORT` / `CLAUDE_EFFORT` | `medium` | Reasoning effort. |
| `CLAUDE_CODE_MODEL` / `CLAUDE_MODEL` | `claude-opus-5-5` | Default model for Claude Code / the API. |
| `ANTHROPIC_API_KEY` | – | Enables the Claude API provider. |
| `OPENAI_API_KEY` | – | Enables the OpenAI API provider. |
| `OPENAI_MODEL` / `OPENAI_EFFORT` | `gpt-6-sol` / `medium` | Default model and reasoning effort for the OpenAI API. |
| `SUMMARY_LANGUAGE` | `English` | Language of the summaries. |
| `WHISPER_DEVICE` | `cpu` | `cpu` or `cuda`. |
| `WHISPER_MODEL` | `small` | Multilingual models only (`small`, `medium`, `large-v3`). |
| `WHISPER_COMPUTE_TYPE` | `int8` | `float16` on a GPU. |
| `WHISPER_CPU_THREADS` | `0` (= 4 threads) | CPU threads for Whisper; raise it on CPUs with more cores. |
| `MAX_DURATION_MIN` | `180` | Longer videos are refused. |
| `WHISPER_RAM_FRACTION` | `0.5` | Share of RAM Whisper may use (checked when a transcription gets its turn; one runs at a time). A video that doesn't fit right now waits while other videos go first; one that could never fit is refused. |
| `WHISPER_RAM_WAIT_MIN` | `60` | How long such a video waits for memory (the user can stop waiting with a button). |
| `MAX_QUEUED_PER_USER` | `3` | Videos one user may have queued or running at once (admins: no limit). |
| `MAX_QUEUE` | `20` | Total videos in the queue; new links are refused beyond that (admins excepted). |
| `YOUTUBE_PARALLEL` | `2` | YouTube lookups (video info, captions) at the same time. A burst of them gets the server's IP flagged ("Sign in to confirm you're not a bot"); downloads from a looked-up video don't count. |
| `WORKERS` | `8` | Jobs worked on at the same time. Lookups, downloads and AI calls run side by side; Whisper, OCR, frame sweeps and voice messages take turns on the CPU. |
| `AGAIN_COOLDOWN_MIN` | `10` | Minutes before the same user may `/again` the same video again (admins: no limit). |
| `MAX_DOC_PAGES` | `2000` | Most pages read from an uploaded document (EPUB/DOCX/TXT count ~2000 characters as a page). |
| `MAX_DOC_CHARS` | `3000000` | Most characters read from an uploaded document. |
| `MAX_MEDIA_LINK_MB` | `1024` | Largest audio/video file downloaded from a Google Drive / Dropbox link (only its audio is decoded; it's deleted after the job). |
| `MAX_LINK_DOWNLOAD_MB` | `100` | Largest document downloaded from a Google Drive / Dropbox link. |
| `TTS_VOICE` | *(empty)* | Kokoro voice for 🔊 voice messages; empty = the default for `SUMMARY_LANGUAGE` (English: `af_heart`). |
| `TTS_SPEED` | `1.0` | Reading speed, 0.5–2.0. |
| `TTS_DEVICE` | `cpu` | `cuda` makes voice messages on an NVIDIA GPU (with `onnxruntime-gpu`, see GPU below): seconds instead of about a minute. Falls back to the CPU without CUDA. |
| `VOICE_CACHE_DAYS` | `7` | Days a made voice message is reused (only Telegram's id for it is stored); then it's made again. |
| `ASK_MAX_CHARS` | `BOOK_CHUNK_CHARS` | Most a question may send to the AI, in characters (summary, earlier answers, transcript or book text). Longer material is cut; a longer book falls back to its summaries. |
| `ASK_MAX_QUESTION` | `1000` | Longest question accepted, in characters. |
| `TTS_DAILY_LIMIT` | `20` | New voice messages a user may have made per 24 h (admins: no limit; `0` = no limit). Reused ones are free, and listening doesn't count toward `DAILY_LIMIT`. `/limit voice` changes it. |
| `ESPEAK_LIB` / `ESPEAK_DATA` | Debian/Ubuntu paths | The system's espeak-ng, which Kokoro needs. |
| `UNIT_SYSTEM` / `TEMPERATURE` | `metric` / `c` | Default units for measurements in summaries; each user can change theirs with `/units`. |
| `OCR_ENGINE` | `auto` | Text recognition for scanned PDFs. `auto`: RapidOCR when `OCR_DEVICE=cuda` works, else Tesseract (measured on a 475-page scanned book, 4 cores, 3 workers: Tesseract 1.2 s/page, RapidOCR 6.4 s/page, same text). Or force `tesseract` / `rapidocr`. Switching engines needs the OCR languages added again (`/ocrlang`); the bot logs which at startup. |
| `OCR_DEVICE` | `cpu` | `cuda` runs OCR with RapidOCR on an NVIDIA GPU (see GPU below); falls back to Tesseract on the CPU when CUDA isn't available. |
| `ASK_DAILY_LIMIT` | `20` | Follow-up questions (💬 Ask) a user may ask per 24 h (admins: no limit; `0` = no limit). They don't count toward `DAILY_LIMIT`. `/limit ask` changes it. |
| `OCR_DAILY_LIMIT` | `5` | Scanned documents a user may have read per rolling 24 h (admins: no limit; `0` = no limit). Once changed with `/limit ocr`, the stored value wins. |
| `OCR_MAX_PAGES` | `400` | Longest scan read without an admin's approval (pages needing OCR). |
| `OCR_WORKERS` | `3` | OCR processes in parallel (one thread each); keep below the number of cores. |
| `BOOK_PARALLEL` | `3` | AI calls for one book that run side by side (chapter batches); the machine mostly waits for the AI meanwhile. `1` runs them one after another. |
| `BOOK_CHUNK_CHARS` | `300000` | Book text sent to the AI in one call (~75k tokens). Several short chapters share a call; longer books and chapters are summarized in pieces, then combined. |
| `DAILY_LIMIT` | `100` | Links a user may send per rolling 24 h; every accepted link counts (`/again`, `/transcript`, cache hits, failures). `0` = no limit; admins: no limit. Only the starting value: once changed with `/limit`, the stored value wins. |
| `MAX_FRAMES` / `MAX_SLIDES` | `16` / `35` | Images sent to the LLM. |
| `SHORT_VIDEO_SEC` | `180` | Up to this length, frames are also sampled every ~2 s. |

### GPU

To run Whisper on an NVIDIA GPU:

1. Install the CUDA libraries into the venv: `.venv/bin/pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`.
2. In `.env`: `WHISPER_DEVICE=cuda`, `WHISPER_COMPUTE_TYPE=float16`, and a larger model such as
   `WHISPER_MODEL=medium` (more accurate, especially for non-English speech).
3. Uncomment the `LD_LIBRARY_PATH` line in `deploy/telegram-summarizer-bot.service` (the CUDA libraries
   must be on the loader path before Python starts), copy the unit again, and restart.
4. Optionally `sudo apt install ffmpeg`; the bot prefers a system ffmpeg over the bundled one.

The bot test-runs the model when it loads and falls back to the CPU if CUDA isn't working.

To run OCR on the GPU too: `.venv/bin/pip uninstall -y onnxruntime && .venv/bin/pip install
onnxruntime-gpu` (faster-whisper works with either), set `OCR_DEVICE=cuda`, and restart; with the default
`OCR_ENGINE=auto` that switches OCR to RapidOCR (add the OCR languages again with `/ocrlang`). Voice messages
use the GPU with `TTS_DEVICE=cuda` (same onnxruntime-gpu). A page then takes
a fraction of a second instead of a few seconds; OCR runs as one process that gets the GPU's devices in its
sandbox. Without a working CUDA runtime it logs a warning and stays on the CPU.

## Commands

| Command | Who | |
|---|---|---|
| *(a link)* | users | Summarize it. |
| `/again <url>` | users | Summarize again with your current model, ignoring the cache. |
| `/transcript <url>` | users | The raw transcript as a `.txt` file (YouTube/TikTok, or a Drive/Dropbox link to a recording; for an uploaded recording, send it with the caption `/transcript`). |
| `/history` | users | Your recent requests (admins: everyone's, with who sent them). |
| `/models` | users | Show or choose the AI: provider first, then model. |
| `/limit` | users | Your limits: requests, scanned documents (OCR) and new voice messages in the last 24 h, and how many are left. |
| `/limit` | admins | Show the daily and OCR limits and per-user overrides. `/limit 50` sets the daily limit for everyone, `/limit <user id> 200` for one user, `/limit <user id> default` removes the override; `0` = no limit. The same with `ocr`, `voice` or `ask` first (`/limit ocr 5`, `/limit voice 20`, `/limit ask 20`) for the OCR, voice-message and question limits. |
| `/ocrlang` | admins | Languages for scanned documents: `/ocrlang` lists them, `/ocrlang add slv` downloads and installs one (Tesseract: from tesseract-ocr's `tessdata_fast` on GitHub, checked with a test run; RapidOCR: the script's model, checked against RapidOCR's pinned SHA-256), `/ocrlang remove slv`. |
| `/units` | users | Units for measurements in summaries: Metric / Imperial and °C / °F (default: metric, °C). |
| `/users` | admins | Users with Allow / Deny / Remove / Unblock buttons, their AI choice and daily limit with today's usage. |
| `/help` | users | How to use the bot (admins also see their commands). |
| `/start` | strangers | Request access. |

The command menu adapts per user: strangers only see `/start`.

### Books and documents

Send a **PDF, EPUB, DOCX or TXT** file (up to 20 MB, the most Telegram lets bots download), or a **Google
Drive or Dropbox link** to one (up to `MAX_LINK_DOWNLOAD_MB`, default 100 MB; shared as "Anyone with the
link"; Google Docs documents work too), and pick:

- **📖 Whole book**: title, author and a summary of the whole thing, with the chapter options below it
  for more detail.
- **📑 By chapter**, then one of:
  - **All chapters, short**: 1–2 paragraphs per chapter, in 1–3 messages.
  - **All chapters, one per message**: a full summary of every chapter.
  - **Pick a chapter**: the chapter list as buttons; a full summary of the chosen one.

Chapters come from the file's table of contents (PDF bookmarks, EPUB contents, DOCX heading styles) or,
failing that, from headings like "Chapter 3" / "Poglavje 3"; very many short chapters are grouped, and text
without any structure is split into parts. Old Word `.doc` and Kindle files must be converted to PDF or
EPUB first; password-protected PDFs and copy-protected (DRM) e-books can't be read.

**Scanned PDFs** (pages that are pictures, no text) need OCR, which is slow on a CPU (about 1.2 s per page
with 3 workers; a 300-page book takes 10–15 minutes) and has its own daily limit (`OCR_DAILY_LIMIT`, default
5). The bot first checks a few pages for the language, then asks: "312 pages need text recognition, about
10 min. Start?" Scans longer than `OCR_MAX_PAGES` (400) can't be started by the user; they can ask an admin,
who gets the details (pages, time, language, the user's OCR use) with Allow / Deny buttons. OCR'd text is
kept like any other, so a scan is read only once. A scan in a language that isn't installed is refused with
the list of supported ones; admins add languages with `/ocrlang`.

Each request counts toward the daily limit (picking a chapter from a list you just requested doesn't count
again). Once a file has been read, "Pick a chapter" shows the list at once, without queueing; only the
chapter you tap is a request. Files are parsed in a sandbox without network access. Only the extracted text is kept (by the file's
SHA-256, like video transcripts), so the same file is read once; summaries are cached per model. Admins see
file names in `/history`.

### Voice messages, audio and video files

Send a **voice message**, an **audio file** or a **video** (also a round video message, or such a file sent as a
document) and it's summarized right away, like a YouTube link: Whisper transcribes it, and for a video the AI can
ask to see frames. The reply has a short title the AI writes and the summary (no clickbait section; a recording
has no published title), with 🔊 Listen and your units. Send it with the caption `/transcript` for the
transcript as a file instead.

- Up to 20 MB through Telegram (the most bots may download); bigger ones as a **Google Drive or Dropbox link**
  (shared as "Anyone with the link"), up to `MAX_MEDIA_LINK_MB` (1 GB). For a link, the bot first reads only
  the file's first bytes to learn its name, type and size, so a recording is summarized right away and a
  document gets the book menu; the size and free disk space are checked before it's queued.
- `/transcript <link>` gives a linked recording's transcript as a file.
- The length limit (`MAX_DURATION_MIN`) and the memory check are applied before anything is downloaded.
- The file is decoded only in the sandbox (a file can be crafted against ffmpeg) and deleted after the job;
  the transcript and summaries are kept by the file's SHA-256, so sending it again is quick. The shared cache
  never holds the file's name: someone sending the same file later doesn't see what you called it.

### Units

Measurements in summaries are shown in each reader's units: metric and °C by default, or imperial / °F.
Summaries keep the video's units in the cache (shared by everyone); the text is converted for each reader
when it's shown, by fixed rules, without the AI. Only the converted value is shown, so only clear cases are
converted: a number in digits right before an unambiguous unit (°F/°C, miles/km, feet/metres, inches,
mph/km/h, lb/kg, gallons/litres, fl oz/ml, sq ft/m²; heights like 6'2" become one value). Things that are
often not measurements stay as written: titles ("500 Miles"), quoted prices, compound units (mpg, lb-ft),
fractions, nominal sizes ("65-inch TV", "3.5 mm jack"), "pounds" without a weight context (a UK price),
airline miles. Voice messages use the converted values, read as words.

### Voice messages (🔊 Listen)

Every summary (videos, books, chapters) has a **🔊 Listen** button. Tapping it makes a Telegram voice
message of what you read: title, clickbait answer, summary (or the book / chapters), without the footer and
links, with bullets read as sentences. The whole summary is read, however long.

- Made with [Kokoro](https://github.com/thewh1teagle/kokoro-onnx) on this machine, in the sandbox: about
  1 minute for a typical summary on the CPU (2.4× faster than real time), seconds with a GPU
  (`TTS_DEVICE=cuda`). The model (338 MB) is downloaded once at startup; until it's ready, there's no button.
- A made voice message is reused for `VOICE_CACHE_DAYS` (7): only Telegram's id for it is stored, no audio
  on disk. Someone hearing a voice message another user had made first gets it after the usual privacy
  pause.
- Listening doesn't count toward the daily limit; new voice messages have their own (`TTS_DAILY_LIMIT`,
  20 per 24 h, `/limit voice`); reused ones are free.
- Read in the language of `SUMMARY_LANGUAGE` (English, Spanish, French, Italian, Portuguese or Hindi; others
  get no button).

### Questions and downloads (💬 Ask, 📄 Download)

Every summary (videos, recordings, books, chapters) also has these buttons:

- **💬 Ask**: ask anything about the summary, e.g. "make it longer" or "what did they say about prices?".
  Tap the button and type the question, or just reply to any part of the summary or to an earlier answer. The
  answer comes as a reply, with its own 💬 to ask on.
  - The AI gets the summary, your last 3 questions about it, and what it came from: the transcript, or a
    book's full text. A book too long to send in one go is answered from its stored summaries plus the
    chapters the question names ("chapter 7", or a chapter's title). Everything stays within
    `ASK_MAX_CHARS`.
  - It answers only from that material and declines unrelated requests (it isn't a general chatbot).
    Questions are capped at `ASK_MAX_QUESTION` characters, answers at 8000.
  - Questions have their own daily limit (`ASK_DAILY_LIMIT`, 20 per 24 h, `/limit ask`) and don't count
    toward the daily request limit; each answer says how many are left today. One open question per summary at
  a time.
  - Only the person who got the summary (or an admin) can ask about it; answers are never shared between
    users. Admins see questions in `/history`, as they see links.
- **📄 Download**: a Markdown file to take to ChatGPT or any other chat. For videos and recordings: the
  summary (in your units) and the timestamped transcript. For books and documents: the full text, with chapter
  headings. Sent right away (no AI, no limit; one file per summary every 10 s). Unlike `/transcript`, it
  doesn't use a request.

## How it works

1. **Link check** as soon as the link arrives: only YouTube/TikTok hosts, no network needed; then the
   daily limit (100 links per 24 h by default) and queue limits (3 per user, 20 in total); admins exempt.
2. **URL** → platform + id (tracking parameters stripped, TikTok short links resolved hop by hop).
3. **Lookup** with yt-dlp, once per job: captions and downloads reuse its answer (`--load-info-json`), so
   the video page isn't fetched again (it is once more only if the stored links expired). A summary written
   from a saved transcript (another model, `/again`) needs no lookup at all. TikTok carousels are detected
   here (no video formats); live streams and videos of unknown length are refused.
4. **Transcript:** YouTube captions → Whisper. TikTok: Whisper → TikTok's captions. Carousels: slides
   via gallery-dl instead. No or hardly any speech → frames are grabbed right away. Before Whisper, the
   memory guard checks the transcription fits in RAM; if not, the job waits while others go first.
5. **LLM, turn 1:** metadata, thumbnail and timestamped transcript → summary, plus whether frames are
   needed and at which moments.
6. **LLM, turn 2 (only if asked):** frames from those moments, in the same conversation (Codex by its
   exact session id, Claude Code by a per-video UUID) → revised summary.
7. **Reply** in Telegram (HTML, split at 4096 characters). Downloaded media and LLM session files are
   deleted after each job (and any leftovers of a crashed run at the next start).

**Books and documents** take the same queue:

1. **Intake:** a file (type and the 20 MB Telegram limit checked) or a Drive/Dropbox link (parsed into the
   service's own download URL); the user picks whole book / chapters. Each pick passes the same limits.
2. **Download** from Telegram or the share link (HTTPS, the service's hosts only, size-capped).
3. **Reading** in a sandbox with no network: the format is told from the bytes; text and chapters come
   from PDF bookmarks, the EPUB contents, DOCX heading styles, or headings in the text.
4. **OCR** for scans, after a language check on a few pages and the user's confirmation; pages are stored
   as they're done.
5. **LLM:** chapters are packed several per call, and up to `BOOK_PARALLEL` calls run side by side; long
   chapters and books are summarized in pieces, then combined. Every summary (and every piece) is cached as
   soon as it exists, and a failed call is retried once before the job fails.
6. **Reply:** Title / Author / Summary, or one block per chapter.

Up to `WORKERS` jobs (8) run side by side, so a captioned video, a cached summary, a 💬 question or a book's
summary never waits behind someone's hour-long transcription. Only the CPU-heavy steps take turns, first
come first served: Whisper, OCR, frame sweeps and Kokoro each use every core (and Whisper a lot of memory).
A job reaching one while another runs shows "⏳ Waiting for a turn on the transcription engine…" with a
rough wait; time spent waiting isn't counted as work in footers or speed estimates. While transcribing,
the status shows Whisper's progress in % and an ETA from this run's own speed. A job with a step of a minute or
more (a long transcription, OCR, a big book, waiting for a turn) gets a ✖️ Cancel button on its status
message; it stops the job's programs at once. A cancelled job, or one stopped by a restart, gets a
🔁 Try again button that queues it again as it was (the same request, so limits aren't charged twice;
it works after a restart because each request stores its job). Two jobs for the same video
or document never run at once: the second waits and then reuses the first one's work. A transcription that
doesn't fit in memory yet waits aside and retries; it may alternate between waiting for memory and waiting
for its turn.

When all workers are busy, the status message shows an estimated wait (never the position, which would
reveal how busy others are); then each stage and an ETA
learned from this machine's measured speeds. Every external program runs in its own process group, so a
timeout, a cancelled job or a removed user stops it at once.

### Code layout

`bot.py` only starts the bot; the Telegram side lives in `tgbot/` (handlers, the job queue and worker, one
runner per job kind, rendering and delivery) and the work itself in `summarizer/` (pipeline, documents,
OCR, speech, LLM calls). AGENTS.md has the module table and the rules for changing them.

## Security and privacy

Video titles, descriptions, transcripts and on-screen text are untrusted: a video can contain prompt
injection. The model therefore gets no capability beyond returning its JSON answer:

| Provider | Isolation |
|---|---|
| Codex | `codex exec` inside `bwrap`: read-only `/usr` and certificates, the bot's own `CODEX_HOME`, and this job's images. No `/home`, `.env` or repo. Every tool feature switched off (shell, JavaScript runtime, image viewing and generation, sub-agents, browser, computer use, apps, plugins…), web search off, Codex's own sandbox read-only. |
| Claude Code | `claude -p --tools ""`: no tools, no MCP servers; images are sent inline. |
| Claude API | A plain Messages call with no tools. |
| OpenAI API | A plain Responses call with no tools. Turn 2 continues server-side via `previous_response_id`; the stored responses are deleted after each job. |

- Uploaded files and linked documents are untrusted too. They're parsed only inside a `bwrap` sandbox with
  no network, no environment and no repository (so no `.env`), under a memory limit; EPUB/DOCX archives
  are read with size and compression-ratio caps (zip bombs). Share links are never fetched as sent: the
  bot builds the download URL from the file id and follows redirects only on Google's / Dropbox's hosts.
  OCR models are downloaded only from fixed sources and checked before use.
- Who submitted what is visible only to admins. Someone requesting a video another user already
  processed can't tell: they see the same stages as a real run, the answer arrives after half the original
  processing time (at most 2 minutes), and the footer's timings match. This covers summaries, `/transcript`
  and summaries written from a saved transcript (the transcript step is shown and paced, too). Missing
  timings (older summaries) are estimated. Only admins and a repeat of a user's own request are answered
  instantly from the cache.
- All users share the admin's subscription limits.
- The bot works only in private chats and leaves any group it's added to.
- Users only see fixed, expected error messages (never raw tool output, paths or stack traces). Admins see
  the technical details on their own requests and get a notice when a user's request fails unexpectedly
  (at most one per error type every 10 minutes).
- Strangers can't flood it: they only get an answer to `/start`, a pending user is reminded at most every
  10 minutes, blocked users are ignored silently, and at most 10 access requests can be pending at once
  (further `/start`s are declined without notifying the admins).

## Data (`data/`, not in git)

`bot.sqlite3` holds these tables:

- `users`: id, name, username, status (admin / allowed / pending / blocked), chosen `backend` + `model`,
  per-user limit overrides (`daily_limit`, `ocr_limit`, `tts_limit`; empty = the global value) and units
  (`unit_system`, `temperature`); `ask_limit` too.
- `videos`: metadata and transcript, status `processing` → `done` / `failed` (or `waiting` for memory,
  `cancelled`).
- `summaries`: one per video and model.
- `requests`: every link or document request: who, what, kind, status (`queued` → `processing` → `done` /
  `failed` / `cancelled`; `waiting` for a chapter pick or an OCR confirmation), cache hit, whether it ran
  OCR, error detail, timestamps.
- `uploads`: files and share links users sent (Telegram file id or the link, name, size, the document).
- `documents` / `document_pages`: a document's metadata and chapters, and its text page by page (from the
  file or OCR), by the file's SHA-256.
- `doc_summaries`: book and chapter summaries, per style and model.
- `ocr_holds`: document requests waiting for an OCR confirmation or an admin's approval.
- `spoken`: per summary request, exactly the text its 🔊 button reads aloud (with the language and voice).
- `voices`: Telegram's id of each made voice message, by a hash of voice, language, speed and text (no
  user), so the same text isn't synthesized twice; deleted after `VOICE_CACHE_DAYS`.
- `delivered`: what each summary (and each 💬 answer) showed, as plain text with the model that wrote it;
  💬 and 📄 work from it.
- `messages`: the bot's summary, answer and 💬 prompt messages per chat, so a reply to any of them is a
  question about the right summary.
- `settings`: values set from Telegram (`/limit`, `/limit ocr`, `/limit voice`, `/limit ask`, `/ocrlang`).

Also there: `codex-home/` (the bot's Codex login), `stats.json` (measured speeds for ETAs), `tessdata/` and
`rapidocr/` (OCR language models added with `/ocrlang`).

## Maintenance

- **Stopping the bot** (`systemctl --user stop`/`restart`) tells everyone still waiting for a summary that
  the bot was stopped and they can retry later, and stops their downloads and transcriptions. The service
  file uses `KillMode=mixed` and `TimeoutStopSec=60` for this; after changing it, copy it again and run
  `systemctl --user daemon-reload`. Requests lost in a crash are marked failed at the next start.
- **Blocks:** if YouTube or TikTok throttle or block the server ("too many requests", "confirm you're not a
  bot"), users are told to try later and the admins get the raw error, at most once per platform every 6 h.
- **yt-dlp / gallery-dl** break when YouTube or TikTok change. The weekly timer runs
  `deploy/update-extractors.sh` to upgrade them; no restart needed.
- **Codex / Claude Code** are installed outside the venv. Every 12 h the bot checks npm for newer
  releases and messages the admins once per version with the update command. New models may need them.
- **OCR:** Tesseract comes from the system (`sudo apt upgrade` keeps it current); its language models are
  in `data/tessdata` and are added or removed with `/ocrlang`. Switching `OCR_ENGINE` needs the languages
  added again for the other engine (RapidOCR has one model per script). RapidOCR updates with the Python
  dependencies.
- **New models:** Codex's list updates itself; the Claude Code list is in `summarizer/summarize.py`
  (`CLAUDE_CODE_MODELS`).

## License

Copyright (C) 2026 Sandi Verdev

This program is free software: you can redistribute it and/or modify it under the terms of the GNU
General Public License as published by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version. It is distributed WITHOUT ANY WARRANTY; see [LICENSE](LICENSE)
for details.
