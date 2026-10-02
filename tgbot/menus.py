"""Inline keyboards built in more than one place (book menus, chapter lists, OCR and unit choices)."""
import re

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from summarizer import db


UNIT_CHOICES = {"sys": {"metric": "Metric", "imperial": "Imperial"}, "temp": {"c": "°C", "f": "°F"}}


def units_menu(uid: int) -> tuple[str, InlineKeyboardMarkup]:
    """The /units message: the current choice and two rows of buttons (✓ on the current ones)."""
    system, temperature = db.get_user_units(uid)
    current = {"sys": system, "temp": temperature}
    rows = [[InlineKeyboardButton(("✓ " if current[group] == value else "") + label,
                                  callback_data=f"units:{group}:{value}")
             for value, label in choices.items()] for group, choices in UNIT_CHOICES.items()]
    text = (f"📏 Units in summaries: {UNIT_CHOICES['sys'][system]}, temperatures in "
            f"{UNIT_CHOICES['temp'][temperature]}.\nChoose below; it applies to new replies.")
    return text, InlineKeyboardMarkup(rows)


BOOK_MODES = {"whole": ("book", "📖 Whole book"), "short": ("chapters-short", "All chapters, short"),
              "each": ("chapters", "All chapters, one per message"), "pick": ("chapter-list", "Pick a chapter")}


CHAPTERS_PER_PAGE = 8


def book_menu(upload_id: int, chapters: bool = False, back: bool = True) -> InlineKeyboardMarkup:
    """The choice buttons under an upload: whole / by chapter, or the three chapter options (with ◀ Back to
    the first choice unless `back` is False, as under a whole-book summary)."""
    if not chapters:
        return InlineKeyboardMarkup([[InlineKeyboardButton("📖 Whole book", callback_data=f"book:{upload_id}:whole"),
                                      InlineKeyboardButton("📑 By chapter",
                                                           callback_data=f"book:{upload_id}:chapters")]])
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("All chapters, short", callback_data=f"book:{upload_id}:short")],
        [InlineKeyboardButton("All chapters, one per message", callback_data=f"book:{upload_id}:each")],
        [InlineKeyboardButton("Pick a chapter", callback_data=f"book:{upload_id}:pick")],
    ] + ([[InlineKeyboardButton("◀ Back", callback_data=f"book:{upload_id}:back")]] if back else []))


def chapter_list(upload_id: int, name: str, chapters: list[dict], page: int) -> tuple[str, InlineKeyboardMarkup]:
    """One page of the chapter list: the text and the buttons (chapters plus ◀ ▶)."""
    pages = max(1, -(-len(chapters) // CHAPTERS_PER_PAGE))
    page = min(max(page, 0), pages - 1)
    first = page * CHAPTERS_PER_PAGE
    rows = [[InlineKeyboardButton(f"{i + 1}. {re.sub(r'\s+', ' ', ch['title'])}"[:60],  # titles come from the file
                                  callback_data=f"book:{upload_id}:ch:{i}")]
            for i, ch in enumerate(chapters[first:first + CHAPTERS_PER_PAGE], first)]
    nav = []
    if page > 0:
        nav.append(InlineKeyboardButton("◀", callback_data=f"book:{upload_id}:pg:{page - 1}"))
    if page < pages - 1:
        nav.append(InlineKeyboardButton("▶", callback_data=f"book:{upload_id}:pg:{page + 1}"))
    if nav:
        rows.append(nav)
    text = f"📑 {name[:80]}: pick a chapter" + (f" (page {page + 1} of {pages})" if pages > 1 else "")
    return text, InlineKeyboardMarkup(rows)


def ocr_buttons(rid: int, over_cap: bool) -> InlineKeyboardMarkup:
    """Start/Cancel under an OCR confirmation, or OK/Ask an admin when the scan is over the page cap."""
    if over_cap:
        return InlineKeyboardMarkup([[InlineKeyboardButton("OK", callback_data=f"ocr:{rid}:ok"),
                                      InlineKeyboardButton("🙋 Ask admin for approval",
                                                           callback_data=f"ocr:{rid}:ask")]])
    return InlineKeyboardMarkup([[InlineKeyboardButton("▶️ Start OCR", callback_data=f"ocr:{rid}:go"),
                                  InlineKeyboardButton("✖️ Cancel", callback_data=f"ocr:{rid}:no")]])
