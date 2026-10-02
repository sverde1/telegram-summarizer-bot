"""Builds small test documents: text PDFs (with bookmarks), scanned PDFs, EPUBs and DOCX files."""
import io
import zipfile
from pathlib import Path


def _pdf_text(s: str) -> str:
    """Escapes text for a PDF string literal."""
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(path: Path, pages: list[str], outline: list[tuple[str, int]] = (), *, title: str = "",
             author: str = "") -> Path:
    """Writes a minimal valid PDF with one text line per page line, and optional bookmarks.

    Args:
        path: Where to write it.
        pages: Text of each page (lines separated by newlines).
        outline: (title, 0-based page) bookmarks, all top-level.
        title: Document title metadata.
        author: Author metadata.
    """
    objs: list[bytes] = []  # object n is objs[n - 1]

    def add(body: str | bytes) -> int:
        """Appends an object, returning its number."""
        objs.append(body.encode("latin-1") if isinstance(body, str) else body)
        return len(objs)

    catalog = add("")  # filled in below
    pages_obj = add("")
    font = add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    page_ids = []
    for text in pages:
        ops = ["BT /F1 12 Tf 72 720 Td 14 TL"] + [f"({_pdf_text(ln)}) Tj T*" for ln in text.split("\n")] + ["ET"]
        stream = "\n".join(ops).encode("latin-1", "replace")
        content = add(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
        page_ids.append(add(f"<< /Type /Page /Parent {pages_obj} 0 R /MediaBox [0 0 612 792] "
                            f"/Resources << /Font << /F1 {font} 0 R >> >> /Contents {content} 0 R >>"))
    objs[pages_obj - 1] = (f"<< /Type /Pages /Kids [{' '.join(f'{p} 0 R' for p in page_ids)}] "
                           f"/Count {len(page_ids)} >>").encode()
    outline_ref = ""
    if outline:
        root = add("")
        items = [add("") for _ in outline]
        for i, ((label, page), num) in enumerate(zip(outline, items)):
            links = (f" /Prev {items[i - 1]} 0 R" if i else "") + (f" /Next {items[i + 1]} 0 R" if i + 1 < len(items) else "")
            objs[num - 1] = (f"<< /Title ({_pdf_text(label)}) /Parent {root} 0 R{links} "
                             f"/Dest [{page_ids[page]} 0 R /Fit] >>").encode("latin-1")
        objs[root - 1] = f"<< /Type /Outlines /First {items[0]} 0 R /Last {items[-1]} 0 R /Count {len(items)} >>".encode()
        outline_ref = f" /Outlines {root} 0 R"
    objs[catalog - 1] = f"<< /Type /Catalog /Pages {pages_obj} 0 R{outline_ref} >>".encode()
    info = add(f"<< /Title ({_pdf_text(title)}) /Author ({_pdf_text(author)}) >>")

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for n, body in enumerate(objs, 1):
        offsets.append(out.tell())
        out.write(b"%d 0 obj\n" % n + body + b"\nendobj\n")
    xref = out.tell()
    out.write(b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1))
    for off in offsets:
        out.write(b"%010d 00000 n \n" % off)
    out.write(f"trailer\n<< /Size {len(objs) + 1} /Root {catalog} 0 R /Info {info} 0 R >>\n"
              f"startxref\n{xref}\n%%EOF\n".encode())
    path.write_bytes(out.getvalue())
    return path


def make_scan(path: Path, pages: list[str]) -> Path:
    """Writes an image-only PDF ("scan") whose pages show the given text as pixels."""
    from PIL import Image, ImageDraw, ImageFont
    images = []
    for text in pages:
        img = Image.new("L", (1240, 1754), 255)  # A4 at 150 dpi
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default(size=36)
        except TypeError:  # old Pillow
            font = ImageFont.load_default()
        draw.multiline_text((100, 100), text, fill=0, font=font, spacing=16)
        images.append(img)
    images[0].save(path, save_all=True, append_images=images[1:], resolution=150)
    return path


def make_epub(path: Path, chapters: list[tuple[str, str]], *, nav: bool = True, title: str = "Book",
              author: str = "Ana", encryption: str | None = None) -> Path:
    """Writes an EPUB 3 with one XHTML file per (title, text) chapter and, optionally, a nav table of contents.

    Args:
        encryption: Contents of META-INF/encryption.xml, if any (to test DRM detection).
    """
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        z.writestr("META-INF/container.xml",
                   '<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" '
                   'version="1.0"><rootfiles><rootfile full-path="OEBPS/content.opf" '
                   'media-type="application/oebps-package+xml"/></rootfiles></container>')
        if encryption:
            z.writestr("META-INF/encryption.xml", encryption)
        items, refs, links = [], [], []
        for i, (ch_title, text) in enumerate(chapters, 1):
            body = "".join(f"<p>{para}</p>" for para in text.split("\n"))
            z.writestr(f"OEBPS/ch{i}.xhtml", f'<html xmlns="http://www.w3.org/1999/xhtml"><head><title>x</title>'
                                             f'<style>p{{}}</style></head><body><h1>{ch_title}</h1>{body}</body></html>')
            items.append(f'<item id="c{i}" href="ch{i}.xhtml" media-type="application/xhtml+xml"/>')
            refs.append(f'<itemref idref="c{i}"/>')
            links.append(f'<li><a href="ch{i}.xhtml#top">{ch_title}</a></li>')
        if nav:
            z.writestr("OEBPS/nav.xhtml", '<html xmlns="http://www.w3.org/1999/xhtml" '
                                          'xmlns:epub="http://www.idpf.org/2007/ops"><body><nav epub:type="toc">'
                                          f'<ol>{"".join(links)}</ol></nav></body></html>')
            items.append('<item id="nav" href="nav.xhtml" properties="nav" media-type="application/xhtml+xml"/>')
        z.writestr("OEBPS/content.opf",
                   '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
                   '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">'
                   f'<dc:title>{title}</dc:title><dc:creator>{author}</dc:creator></metadata>'
                   f'<manifest>{"".join(items)}</manifest><spine>{"".join(refs)}</spine></package>')
    return path


def make_docx(path: Path, paragraphs: list[tuple[str, str]], *, title: str = "") -> Path:
    """Writes a DOCX from (style, text) paragraphs, style "" for normal text or e.g. "Heading1"."""
    w = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    body = "".join(
        f'<w:p>{f"<w:pPr><w:pStyle w:val=\"{style}\"/></w:pPr>" if style else ""}<w:r><w:t>{text}</w:t></w:r></w:p>'
        for style, text in paragraphs)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        z.writestr("word/document.xml", f'<w:document xmlns:w="{w}"><w:body>{body}</w:body></w:document>')
        z.writestr("docProps/core.xml", '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/'
                                        '2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/">'
                                        f'<dc:title>{title}</dc:title></cp:coreProperties>')
    return path
