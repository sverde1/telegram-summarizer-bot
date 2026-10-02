"""The Telegram side of the bot (bot.py wires it up). Inside the package, import modules, never names:
tests patch module attributes, and a name imported with `from x import y` would keep the original."""
import os

# Opt in to PTB's coming behavior: RetryAfter.retry_after as a timedelta (an int with a deprecation warning
# until then). Must be set before telegram is imported, hence here; sending.retry_seconds handles both.
os.environ.setdefault("PTB_TIMEDELTA", "1")
