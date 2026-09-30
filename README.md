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
4. Frame heuristic (no speech, "as you can see…", numbers in the title missing from speech):
   downloads the video, grabs targeted + evenly spaced frames, drops near-duplicates.
5. Claude (`claude-opus-5-5`) gets metadata, thumbnail, transcript and frames; returns structured JSON.
6. Reply in Telegram (HTML, split at 4096 chars). Downloaded media is deleted after each job.

## Setup
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # fill in TELEGRAM_BOT_TOKEN, ANTHROPIC_API_KEY
.venv/bin/python bot.py
```
With `ALLOWED_USER_IDS` empty the bot replies with your user id; put it in `.env` and restart.
No system packages are needed: ffmpeg comes from `imageio-ffmpeg` (a system ffmpeg is preferred if
present) and deno (yt-dlp's YouTube JS runtime) from pip.

Long polling needs the bot to have no webhook set (`deleteWebhook`).

Run it permanently with the systemd user unit in `deploy/` (instructions inside).

## Commands
`/frames <url>` force frames · `/noframes <url>` skip frames · `/again <url>` re-summarize ignoring
the cached summary · `/transcript <url>` raw transcript as .txt. Adding `frames` / `noframes` after a
link in a normal message works too.

## GPU later
Set `WHISPER_DEVICE=cuda`, `WHISPER_COMPUTE_TYPE=float16`, `WHISPER_MODEL=medium`, install
`nvidia-cublas-cu12 nvidia-cudnn-cu12`, and uncomment `LD_LIBRARY_PATH` in the service unit. The bot
runs a 1-second test inference at model load and falls back to CPU if CUDA is broken.

## Maintenance
Platforms break extractors often: `.venv/bin/pip install -U "yt-dlp[default,curl-cffi]" gallery-dl`.
