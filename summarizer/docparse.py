"""Reads uploaded documents (PDF, EPUB, DOCX, TXT): text, title/author, and the chapter structure.

The input is untrusted, so this module runs inside the sandbox (`python -m summarizer.docparse <in> <out>
<max pages> <max chars>`, see documents.extract) and imports nothing from the bot: summarizer.config would
load `.env` and create directories, neither possible nor wanted in there. The parsing functions are pure
(path or bytes in, dict out), so tests call them directly.

Every format becomes the same shape: a list of "pages" (real pages for PDFs; for the others, text cut into
about PAGE_CHARS characters, with a new page at every chapter start) and chapters as page ranges.
"""
import html
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath

PAGE_CHARS = 2000         # size of a "page" for formats without pages (about a printed page)
GROUP_CHARS = 40_000      # target size when merging many small chapters, or splitting a text without any
MAX_CHAPTERS = 40         # more than this is a list nobody can pick from: small ones get merged
MIN_FRONT_MATTER = 3000   # text before the first chapter longer than this becomes its own "Beginning" part
TITLE_CHARS = 120

# Zip guards for EPUB/DOCX: a zip's stated sizes can lie, so members are read with hard caps instead.
ZIP_MAX_ENTRIES = 5000
ZIP_MAX_TOTAL = 200 * 1024 ** 2   # bytes decompressed over the whole file
ZIP_MAX_RATIO = 200               # a 1 KB member that inflates to over 200 KB is a bomb, not a book

# Chapter headings in text without a table of contents. Words in the languages the bot's users read.
_HEADING_WORDS = re.compile(
    r"^\s*(?:chapter|poglavje|kapitel|chapitre|cap[ií]tulo|capitolo|rozdzia[lł]|poglavlje|glava|глава|part|"
    r"del|teil|partie|book|knjiga)\s+(?:\d{1,3}|[ivxlcdm]{1,8}|[a-zčšžćđ]{3,12})\b[.:\-–—]?\s*(.{0,80})$",
    re.I)
_HEADING_NUMBER = re.compile(r"^\s*(\d{1,3}|[IVXLC]{1,7})\.?\s*$")  # "12" or "XII" alone on a line
_ROMAN = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}


class DocError(Exception):
    """The document can't be read; `code` says why (encrypted, drm, too_many_pages, too_large, empty_zip,
    zip_bomb, broken, unsupported). The bot turns the code into a fixed message."""

    def __init__(self, code: str, detail: str = ""):
        """Stores the reason code and a technical detail for the admins."""
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


# ---------- format detection ----------

def sniff(path: Path) -> str:
    """Tells the format from the file's bytes; the name and the stated type aren't trusted.

    Returns:
        "pdf", "epub", "docx" or "txt".

    Raises:
        DocError: "unsupported" for anything else (images, archives, old .doc, Kindle files...).
    """
    head = path.read_bytes()[:8192]
    if head.startswith(b"%PDF-"):
        return "pdf"
    if head.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(path) as z:
                names = set(z.namelist()[:ZIP_MAX_ENTRIES + 1])
                if "mimetype" in names and _read(z, "mimetype", [0]).strip() == b"application/epub+zip":
                    return "epub"
                if "word/document.xml" in names:
                    return "docx"
        except (zipfile.BadZipFile, OSError):
            raise DocError("broken", "bad zip")
        raise DocError("unsupported", "zip without EPUB/DOCX content")
    if b"\x00" not in head and _decode(head[:4096], strict=True) is not None:
        return "txt"
    raise DocError("unsupported", "unknown file type")


# ---------- shared helpers ----------

def _decode(data: bytes, strict: bool = False) -> str | None:
    """Decodes text: UTF-8, then Windows-1250 (Central European), then Latin-1.

    Args:
        data: The raw bytes.
        strict: Return None instead of falling back to Latin-1 (which accepts any bytes); used to tell
            text files from binary ones. A cut-off multi-byte character at the end is tolerated.
    """
    for enc in ("utf-8-sig", "cp1250"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError as e:
            if enc == "utf-8-sig" and strict and e.start >= len(data) - 3:
                return data[:e.start].decode(enc)  # the sample cut a character in half
    return None if strict else data.decode("latin-1")


def _clean_title(text: str) -> str:
    """One-line, trimmed chapter title."""
    return re.sub(r"\s+", " ", text or "").strip()[:TITLE_CHARS]


def _read(z: zipfile.ZipFile, name: str, budget: list[int]) -> bytes:
    """Reads one zip member with hard limits, whatever sizes the zip claims.

    Args:
        z: The open zip.
        name: Member name.
        budget: One-element list with the bytes decompressed so far; updated (shared across members).

    Raises:
        DocError: "zip_bomb" when the total or the member's compression ratio is implausible.
    """
    info = z.getinfo(name)
    out = bytearray()
    with z.open(info) as f:
        while chunk := f.read(65536):
            out += chunk
            budget[0] += len(chunk)
            if budget[0] > ZIP_MAX_TOTAL:
                raise DocError("zip_bomb", "too much data")
            if len(out) > 1024 ** 2 and len(out) > ZIP_MAX_RATIO * max(info.compress_size, 1):
                raise DocError("zip_bomb", f"{name} inflates too much")
    return bytes(out)


def _open_zip(path: Path) -> zipfile.ZipFile:
    """Opens an EPUB/DOCX, refusing archives with absurd entry counts.

    Raises:
        DocError: "broken" for an unreadable zip, "zip_bomb" for too many entries.
    """
    try:
        z = zipfile.ZipFile(path)
    except (zipfile.BadZipFile, OSError) as e:
        raise DocError("broken", str(e))
    if len(z.infolist()) > ZIP_MAX_ENTRIES:
        z.close()
        raise DocError("zip_bomb", "too many entries")
    return z


def _xml(data: bytes) -> ET.Element:
    """Parses XML from a document. Python's expat refuses entity-expansion bombs.

    Raises:
        DocError: "broken" for invalid XML.
    """
    try:
        return ET.fromstring(data)
    except ET.ParseError as e:
        raise DocError("broken", f"bad XML: {e}")


def _paginate(blocks: list[tuple[str, bool, str]]) -> tuple[list[str], list[tuple[str, int]]]:
    """Cuts flowing text into pages of about PAGE_CHARS, starting a new page at every chapter heading.

    Args:
        blocks: (text, starts_chapter, chapter title) per paragraph, in reading order.

    Returns:
        (pages, chapter starts as (title, page index)).
    """
    pages, current, size, starts = [], [], 0, []
    for text, is_heading, title in blocks:
        if is_heading and current:
            pages.append("\n".join(current))
            current, size = [], 0
        if is_heading:
            starts.append((title or text, len(pages)))
        while len(text) > PAGE_CHARS * 2:  # one huge paragraph: cut it so pages stay bounded
            cut = text.rfind(" ", 0, PAGE_CHARS) if " " in text[:PAGE_CHARS] else PAGE_CHARS
            current.append(text[:cut])
            pages.append("\n".join(current))
            current, size, text = [], 0, text[cut:].lstrip()
        current.append(text)
        size += len(text) + 1
        if size >= PAGE_CHARS:
            pages.append("\n".join(current))
            current, size = [], 0
    if current:
        pages.append("\n".join(current))
    return pages, starts


def _roman(s: str) -> int | None:
    """Value of a Roman numeral (I…CC range), or None if it isn't one."""
    s = s.upper()
    if not s or any(c not in _ROMAN for c in s):
        return None
    total = 0
    for a, b in zip(s, s[1:] + " "):
        v = _ROMAN[a]
        total += -v if b in _ROMAN and _ROMAN[b] > v else v
    return total


def _heading_blocks(lines: list[str], allow_numbers: bool) -> list[tuple[str, bool, str]]:
    """Marks lines that look like chapter headings (no table of contents available).

    Lines like "Chapter 3" or "Poglavje III: Title" always count. Bare numbers ("12", "XII") count only
    when allowed and only as an increasing run (1, 2, 3…), so page numbers and list items don't.
    """
    marks = {}
    for i, line in enumerate(lines):
        if _HEADING_WORDS.match(line) and len(line) < 120:
            marks[i] = line.strip()
    if allow_numbers and len(marks) < 2:
        numbered = []
        for i, line in enumerate(lines):
            if m := _HEADING_NUMBER.match(line):
                token = m.group(1)
                value = int(token) if token.isdigit() else _roman(token)
                if value:
                    numbered.append((i, value))
        run, best = [], []
        for i, value in numbered:  # longest run counting up by one, starting at 1
            if value == 1:
                run = [(i, value)]
            elif run and value == run[-1][1] + 1:
                run.append((i, value))
            if len(run) > len(best):
                best = list(run)
        if len(best) >= 3:
            for i, _ in best:
                following = next((ln.strip() for ln in lines[i + 1:i + 3] if ln.strip()), "")
                marks[i] = f"{lines[i].strip()} {following}"[:TITLE_CHARS] if len(following) < 80 else lines[i]
    return [(line, i in marks, _clean_title(marks.get(i, ""))) for i, line in enumerate(lines)]


def _chapters(pages: list[str], starts: list[tuple[str, int]], is_pdf: bool) -> list[dict]:
    """Turns chapter starts into page ranges, merging or inventing chapters so the list is usable.

    Starts are sorted and de-duplicated; text before the first chapter is its own part when substantial;
    more than MAX_CHAPTERS chapters are merged into groups of about GROUP_CHARS; no usable chapters means
    equal parts of about GROUP_CHARS.

    Returns:
        [{"title", "start", "end"}] with page indices (end exclusive), covering every page.
    """
    n = len(pages)
    if not n:
        return []
    seen, clean = set(), []
    for title, start in sorted(starts, key=lambda s: s[1]):
        if 0 <= start < n and start not in seen:
            seen.add(start)
            clean.append((_clean_title(title) or f"Chapter {len(clean) + 1}", start))
    chars = [len(p) for p in pages]
    if len(clean) >= 2:
        if clean[0][1] > 0:
            if sum(chars[:clean[0][1]]) > MIN_FRONT_MATTER:
                clean.insert(0, ("Beginning", 0))
            else:
                clean[0] = (clean[0][0], 0)
        chapters = [{"title": t, "start": s, "end": e}
                    for (t, s), (_, e) in zip(clean, clean[1:] + [("", n)])]
        if len(chapters) <= MAX_CHAPTERS:
            return chapters
        groups, current = [], []
        for ch in chapters:
            current.append(ch)
            if sum(chars[current[0]["start"]:ch["end"]]) >= GROUP_CHARS:
                groups.append(current)
                current = []
        if current:
            groups.append(current)
        return [{"title": g[0]["title"] if len(g) == 1 else _clean_title(f"{g[0]['title']} – {g[-1]['title']}"),
                 "start": g[0]["start"], "end": g[-1]["end"]} for g in groups]
    # No usable structure: equal parts.
    total = sum(chars)
    parts = max(1, round(total / GROUP_CHARS))
    bounds, acc, target = [0], 0, total / parts
    for i, c in enumerate(chars):
        acc += c
        if acc >= target * len(bounds) and i + 1 < n and len(bounds) < parts:
            bounds.append(i + 1)
    bounds.append(n)
    out = []
    for k, (s, e) in enumerate(zip(bounds, bounds[1:]), 1):
        label = f"Part {k} (pages {s + 1}–{e})" if is_pdf else f"Part {k}"
        out.append({"title": label, "start": s, "end": e})
    return out


def _result(fmt: str, pages: list[str], starts, title="", author="", *, max_pages: int, max_chars: int,
            real_pages: bool = False) -> dict:
    """Checks the limits and builds the parse result shared by all formats.

    Raises:
        DocError: "too_many_pages" / "too_large" over the limits.
    """
    if len(pages) > max_pages:
        raise DocError("too_many_pages", str(len(pages)))
    if sum(len(p) for p in pages) > max_chars:
        raise DocError("too_large", str(sum(len(p) for p in pages)))
    return {"format": fmt, "title": _clean_title(title), "author": _clean_title(author), "pages": pages,
            "chapters": _chapters(pages, starts, real_pages)}


# ---------- PDF ----------

def _pdfinfo(path: Path) -> dict:
    """Runs poppler's pdfinfo; returns its "Key: value" lines as a dict ({} if it fails)."""
    try:
        out = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True, errors="replace",
                             timeout=120).stdout
    except (OSError, subprocess.TimeoutExpired):
        return {}
    return {k.strip(): v.strip() for k, _, v in (line.partition(":") for line in out.splitlines()) if v}


def _pdf_outline(path: Path, n_pages: int) -> list[tuple[str, int]]:
    """Top-level chapter starts from the PDF's bookmarks (outline), or [] if there are none usable.

    pypdf's outline is a nested list; broken destinations are common, so each entry is tried on its own.
    The top level is used, unless it has fewer than two entries and the next level has more.
    """
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            reader.decrypt("")  # owner-password-only PDFs open with an empty user password
        outline = reader.outline
    except Exception:  # noqa: BLE001  (any pypdf failure just means: no bookmarks)
        return []

    def level(items) -> tuple[list[tuple[str, int]], list]:
        """Entries at this level, and the nested lists below it."""
        found, nested = [], []
        for item in items:
            if isinstance(item, list):
                nested.append(item)
                continue
            try:
                page = reader.get_destination_page_number(item)
                if page is not None and 0 <= page < n_pages:
                    found.append((str(item.title), page))
            except Exception:  # noqa: BLE001
                continue
        return found, nested

    top, nested = level(outline)
    if len(top) < 2:
        deeper = [entry for sub in nested for entry in level(sub)[0]]
        if len(deeper) > len(top):
            top = deeper
    return top


def parse_pdf(path: Path, max_pages: int, max_chars: int) -> dict:
    """Text per page with poppler, chapters from bookmarks or headings; also how much text each page has.

    The per-page letter count lets the bot tell scans (pages without text) from text PDFs.

    Raises:
        DocError: encrypted, broken or over the limits.
    """
    info = _pdfinfo(path)
    pages_stated = int(info.get("Pages", "0") or 0)
    if pages_stated > max_pages:
        raise DocError("too_many_pages", str(pages_stated))
    run = subprocess.run(["pdftotext", "-enc", "UTF-8", str(path), "-"], capture_output=True, timeout=600)
    if run.returncode != 0:
        err = run.stderr.decode("utf-8", "replace")[:300]
        raise DocError("encrypted" if "password" in err.lower() or info.get("Encrypted", "").startswith("yes")
                       else "broken", err)
    pages = run.stdout.decode("utf-8", "replace").split("\f")
    if pages and not pages[-1].strip():
        pages.pop()  # poppler ends the last page with a form feed too
    starts = _pdf_outline(path, len(pages))
    if len(starts) < 2:  # no bookmarks: look for headings near the top of each page
        starts = []
        for i, page in enumerate(pages):
            top = [ln for ln in page.splitlines() if ln.strip()][:3]
            for line in top:
                if _HEADING_WORDS.match(line) and len(line) < 120:
                    starts.append((line, i))
                    break
    result = _result("pdf", pages, starts, info.get("Title", ""), info.get("Author", ""),
                     max_pages=max_pages, max_chars=max_chars, real_pages=True)
    result["letters"] = [sum(c.isalpha() for c in p) for p in pages]
    return result


# ---------- EPUB ----------

class _Text(HTMLParser):
    """Collects the readable text of an XHTML chapter, one block element per line."""

    BLOCKS = {"p", "div", "br", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "blockquote", "section"}

    def __init__(self):
        """Starts empty."""
        super().__init__(convert_charrefs=True)
        self.parts, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        """Skips scripts and styles; starts a new line at block elements."""
        if tag in ("script", "style", "head"):
            self.skip += 1
        elif tag in self.BLOCKS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        """Ends a skipped element or a block."""
        if tag in ("script", "style", "head"):
            self.skip = max(0, self.skip - 1)
        elif tag in self.BLOCKS:
            self.parts.append("\n")

    def handle_data(self, data):
        """Keeps text outside skipped elements."""
        if not self.skip:
            self.parts.append(data)

    def text(self) -> str:
        """The collected text, whitespace tidied."""
        lines = (re.sub(r"[ \t\r\f\v]+", " ", ln).strip() for ln in "".join(self.parts).split("\n"))
        return "\n".join(ln for ln in lines if ln)


def _xhtml_text(data: bytes) -> str:
    """Readable text of one XHTML document."""
    p = _Text()
    p.feed(_decode(data) or "")
    p.close()
    return p.text()


def _resolve(base: str, href: str) -> str:
    """A zip path for `href` relative to the document at `base`, without its #fragment; "" if it escapes."""
    href = html.unescape(href.split("#", 1)[0])
    if not href or "://" in href or href.startswith("/"):
        return ""
    parts = []
    for part in (PurePosixPath(base).parent / href).parts:
        if part == "..":
            if not parts:
                return ""
            parts.pop()
        elif part != ".":
            parts.append(part)
    return "/".join(parts)


def _local(tag: str) -> str:
    """XML tag name without its namespace."""
    return tag.rsplit("}", 1)[-1]


def parse_epub(path: Path, max_pages: int, max_chars: int) -> dict:
    """Text in reading order (the spine) and chapters from the table of contents (EPUB 3 nav or EPUB 2 NCX).

    Raises:
        DocError: drm (encrypted content), broken, zip_bomb or over the limits.
    """
    budget = [0]
    with _open_zip(path) as z:
        names = set(z.namelist())
        if "META-INF/encryption.xml" in names:
            # Font obfuscation also lists itself here; only encrypted text means DRM.
            enc = _decode(_read(z, "META-INF/encryption.xml", budget)) or ""
            if re.search(r'URI="[^"]+\.(x?html?|xml)"', enc, re.I):
                raise DocError("drm")
        if "META-INF/container.xml" not in names:
            raise DocError("broken", "no container.xml")
        container = _xml(_read(z, "META-INF/container.xml", budget))
        rootfile = next((el.get("full-path") for el in container.iter() if _local(el.tag) == "rootfile"), None)
        if not rootfile or rootfile not in names:
            raise DocError("broken", "no package document")
        opf = _xml(_read(z, rootfile, budget))
        manifest, spine, title, author, nav, ncx = {}, [], "", "", None, None
        for el in opf.iter():
            tag = _local(el.tag)
            if tag == "item":
                href = _resolve(rootfile, el.get("href", ""))
                manifest[el.get("id")] = href
                if "nav" in (el.get("properties") or "").split():
                    nav = href
                if el.get("media-type") == "application/x-dtbncx+xml":
                    ncx = href
            elif tag == "itemref":
                spine.append(el.get("idref"))
            elif tag == "title" and not title:
                title = el.text or ""
            elif tag == "creator" and not author:
                author = el.text or ""
        docs = [manifest[i] for i in spine if manifest.get(i) in names]
        if not docs:
            raise DocError("broken", "empty spine")

        toc: list[tuple[str, str]] = []  # (title, zip path of the document it points to)
        if nav and nav in names:
            root = _xml(_read(z, nav, budget))
            for el in root.iter():
                if _local(el.tag) == "nav" and "toc" in (el.get("{http://www.idpf.org/2007/ops}type") or "toc"):
                    # Top-level entries only: the <ol> directly inside this <nav>.
                    ol = next((c for c in el if _local(c.tag) == "ol"), None)
                    for li in (ol if ol is not None else []):
                        a = next((c for c in li.iter() if _local(c.tag) == "a"), None)
                        if a is not None and a.get("href"):
                            toc.append(("".join(a.itertext()), _resolve(nav, a.get("href"))))
                    break
        elif ncx and ncx in names:
            root = _xml(_read(z, ncx, budget))
            navmap = next((el for el in root.iter() if _local(el.tag) == "navMap"), None)
            for point in (navmap if navmap is not None else []):
                if _local(point.tag) != "navPoint":
                    continue
                label = next((el for el in point.iter() if _local(el.tag) == "text"), None)
                content = next((el for el in point.iter() if _local(el.tag) == "content"), None)
                if content is not None and content.get("src"):
                    toc.append(((label.text if label is not None else "") or "", _resolve(ncx, content.get("src"))))

        first_entry = {}
        for entry_title, target in toc:
            first_entry.setdefault(target, entry_title)
        blocks = []
        for doc in docs:
            text = _xhtml_text(_read(z, doc, budget))
            lines = [ln for ln in text.split("\n") if ln.strip()]
            if not lines:
                continue
            chapter = doc in first_entry
            blocks.append((lines[0], chapter, _clean_title(first_entry.get(doc, ""))))
            blocks += [(ln, False, "") for ln in lines[1:]]
    if not toc:  # no table of contents: look for headings in the text
        blocks = _heading_blocks([b[0] for b in blocks], allow_numbers=True)
    pages, starts = _paginate(blocks)
    return _result("epub", pages, starts, title, author, max_pages=max_pages, max_chars=max_chars)


# ---------- DOCX ----------

_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def parse_docx(path: Path, max_pages: int, max_chars: int) -> dict:
    """Paragraph text, chapters at "Title"/"Heading 1" paragraphs (or recognisable headings).

    Read with the standard library instead of python-docx, so every byte goes through the zip guards.

    Raises:
        DocError: broken, zip_bomb or over the limits.
    """
    budget = [0]
    with _open_zip(path) as z:
        names = set(z.namelist())
        body = _xml(_read(z, "word/document.xml", budget))
        title = author = ""
        if "docProps/core.xml" in names:
            for el in _xml(_read(z, "docProps/core.xml", budget)).iter():
                if _local(el.tag) == "title" and el.text:
                    title = el.text
                elif _local(el.tag) == "creator" and el.text:
                    author = el.text
    blocks, styled = [], 0
    for p in body.iter(f"{_W}p"):
        text = "".join(t.text or "" for t in p.iter(f"{_W}t")).strip()
        if not text:
            continue
        style = p.find(f"{_W}pPr/{_W}pStyle")
        name = (style.get(f"{_W}val") or "").lower().replace(" ", "") if style is not None else ""
        heading = name in ("heading1", "title", "naslov1", "berschrift1") or name.endswith("heading1")
        styled += heading
        blocks.append((text, heading, _clean_title(text) if heading else ""))
    if styled < 2:  # no heading styles used: look for headings in the text
        blocks = _heading_blocks([b[0] for b in blocks], allow_numbers=True)
    pages, starts = _paginate(blocks)
    return _result("docx", pages, starts, title, author, max_pages=max_pages, max_chars=max_chars)


# ---------- TXT ----------

def parse_txt(path: Path, max_pages: int, max_chars: int) -> dict:
    """Plain text; chapters from recognisable headings.

    Raises:
        DocError: over the limits.
    """
    data = path.read_bytes()
    if len(data) > max_chars * 4:  # even at 4 bytes per character it couldn't fit
        raise DocError("too_large", str(len(data)))
    lines = [ln.rstrip() for ln in (_decode(data) or "").splitlines()]
    blocks = [b for b in _heading_blocks(lines, allow_numbers=True) if b[0].strip() or b[1]]
    pages, starts = _paginate(blocks)
    return _result("txt", pages, starts, max_pages=max_pages, max_chars=max_chars)


PARSERS = {"pdf": parse_pdf, "epub": parse_epub, "docx": parse_docx, "txt": parse_txt}


def parse(path: Path, max_pages: int, max_chars: int) -> dict:
    """Detects the format and parses the file.

    Raises:
        DocError: The file can't be used (see the codes on DocError).
    """
    return PARSERS[sniff(path)](path, max_pages, max_chars)


def main(argv: list[str]) -> int:
    """Sandbox entry point: parses argv[0], writes the result (or {"error": code}) as JSON to argv[1]."""
    src, out, max_pages, max_chars = Path(argv[0]), Path(argv[1]), int(argv[2]), int(argv[3])
    try:
        result = parse(src, max_pages, max_chars)
    except DocError as e:
        result = {"error": e.code, "detail": e.detail[:500]}
    except (RecursionError, MemoryError, ValueError, KeyError, IndexError, OSError) as e:
        result = {"error": "broken", "detail": f"{type(e).__name__}: {e}"[:500]}
    out.write_text(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
