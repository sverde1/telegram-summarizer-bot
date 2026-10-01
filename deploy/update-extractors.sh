#!/usr/bin/env bash
# Upgrade the downloaders that YouTube/TikTok break most often. The bot runs yt-dlp/gallery-dl as
# subprocesses, so new versions take effect on the next video without restarting the bot.
set -euo pipefail
cd "$(dirname "$0")/.."
pip=.venv/bin/pip
# Snapshot versions before and after, so the systemd journal says exactly what changed.
# yt-dlp-ejs and deno: yt-dlp's YouTube JavaScript challenge solver and the runtime it needs;
# curl_cffi: browser impersonation, which TikTok extraction relies on. They break together with yt-dlp.
before=$($pip freeze | grep -iE '^(yt-dlp|yt-dlp-ejs|gallery.dl|deno|curl.cffi)==' | sort)
$pip install --quiet --upgrade --disable-pip-version-check "yt-dlp[default,curl-cffi]" yt-dlp-ejs gallery-dl deno
after=$($pip freeze | grep -iE '^(yt-dlp|yt-dlp-ejs|gallery.dl|deno|curl.cffi)==' | sort)
if [ "$before" = "$after" ]; then
  echo "already up to date"
else
  # `|| true`: grep exits 1 when there is no line to show, which set -e would treat as a failure.
  echo "updated:"; diff <(echo "$before") <(echo "$after") | grep '^[<>]' || true
fi
.venv/bin/yt-dlp --version >/dev/null  # fail loudly if the new build is broken
