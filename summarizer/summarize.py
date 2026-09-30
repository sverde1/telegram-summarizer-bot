"""The Claude call: transcript + thumbnail (+ frames) in, {title, clickbait_answer, summary} out."""
import base64
import json
import logging
from pathlib import Path

import anthropic

from . import config

log = logging.getLogger(__name__)

_client: anthropic.Anthropic | None = None

SYSTEM = f"""You summarize videos for one reader who wants to know quickly what a video actually says,
and whether its title and thumbnail tell the truth.

You get the video's metadata, its thumbnail, a transcript (maybe machine-generated) and sometimes frames
or slide images. Write everything in {config.SUMMARY_LANGUAGE}. Output fields:

title: the video's title as given. If it isn't in {config.SUMMARY_LANGUAGE}, give the original followed by a
  translation in parentheses. For TikTok, where the "title" is the post caption, drop hashtags and keep it short.

is_clickbait: true if the title or thumbnail teases, withholds, or exaggerates something to get the click
  ("You won't believe...", "This changed everything", "The truth about X", a question it doesn't answer,
  a shocked face with an arrow, a number or claim the video doesn't back up).

clickbait_answer: if is_clickbait, ONE paragraph (max ~500 characters) that directly answers the teaser:
  what the thing actually is, what happened, or whether the claim holds up, based on what the video shows.
  Lead with the answer itself. Empty string if not clickbait.

summary: plain text, no Markdown (no *, #, or links syntax). Structure it as:
  a 2-3 sentence overview, then the key points as lines starting with "• ", quoting numbers exactly as
  given. If frames/slides show important information that isn't spoken (tables, settings, results), include it
  and say it was shown on screen. End with one line of caveats if relevant (sponsorships, paid courses,
  "comment X to get it", missing evidence) - only what the material supports.
  Keep the summary under 2500 characters; scale it to the video: a 30-second clip needs a few lines,
  an hour-long talk needs the full budget.

Never invent numbers or details. If the transcript is missing, garbled, or clearly mistranscribed, say so
and rely on what the images show. If something is unclear, say it is unclear."""

SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "is_clickbait": {"type": "boolean"},
        "clickbait_answer": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": ["title", "is_clickbait", "clickbait_answer", "summary"],
    "additionalProperties": False,
}


class SummaryError(RuntimeError):
    pass


def _image(path: Path) -> dict:
    data = base64.standard_b64encode(path.read_bytes()).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


def format_transcript(cues: list[tuple[float, str]], every: float = 30.0) -> str:
    """Join cues into paragraphs with a [m:ss] marker roughly every `every` seconds."""
    out, next_mark = [], 0.0
    for start, text in cues:
        if start >= next_mark:
            m, s = divmod(int(start), 60)
            out.append(f"\n[{m}:{s:02d}] ")
            next_mark = start + every
        out.append(text + " ")
    return "".join(out).strip()


def summarize(meta: dict, platform: str, transcript: str, transcript_source: str, language: str,
              thumbnail: Path | None, images: list[tuple[Path, str]]) -> dict:
    content: list[dict] = []
    if thumbnail:
        content += [{"type": "text", "text": "Thumbnail:"}, _image(thumbnail)]
    for path, label in images:
        content += [{"type": "text", "text": f"Image ({label}):"}, _image(path)]
    content.append({"type": "text", "text": (
        f"Platform: {platform}\nTitle: {meta['title']}\nUploader: {meta['uploader']}\n"
        f"Uploaded: {meta['upload_date']}\nDuration: {meta['duration']} s\n"
        f"Description:\n{meta['description'] or '(none)'}\n\n"
        f"Transcript source: {transcript_source} (language: {language or 'unknown'})\n"
        f"<transcript>\n{transcript or '(no speech found)'}\n</transcript>")})

    global _client
    try:
        _client = _client or anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        with _client.beta.messages.stream(
            model=config.CLAUDE_MODEL,
            max_tokens=16000,
            system=SYSTEM,
            thinking={"type": "adaptive"},
            output_config={"effort": config.CLAUDE_EFFORT,
                           "format": {"type": "json_schema", "schema": SCHEMA}},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",  # a safety-classifier refusal is retried on a fallback model server-side
            messages=[{"role": "user", "content": content}],
        ) as stream:
            msg = stream.get_final_message()
    except anthropic.AuthenticationError:
        raise SummaryError("Anthropic API key is missing or invalid (ANTHROPIC_API_KEY in .env).")
    except anthropic.BadRequestError as e:
        raise SummaryError(f"Claude rejected the request: {e.message}")
    except anthropic.RateLimitError:
        raise SummaryError("Claude rate limit hit; try again in a minute.")
    except anthropic.APIStatusError as e:
        raise SummaryError(f"Claude API error {e.status_code}; try again later.")
    except anthropic.APIConnectionError:
        raise SummaryError("Couldn't reach the Claude API (network error).")
    except anthropic.AnthropicError as e:  # e.g. no credentials configured at all
        raise SummaryError(f"Claude client error: {e}")

    u = msg.usage
    log.info("claude %s: in=%s out=%s stop=%s req=%s", msg.model, u.input_tokens, u.output_tokens,
             msg.stop_reason, msg._request_id)
    if msg.stop_reason == "refusal":
        raise SummaryError("Claude declined to summarize this video.")
    if msg.stop_reason == "max_tokens":
        raise SummaryError("Claude's answer was cut off (max_tokens).")
    text = "".join(b.text for b in msg.content if b.type == "text")
    return json.loads(text)
