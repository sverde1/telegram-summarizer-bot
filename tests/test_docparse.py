"""Reading uploaded documents: format detection, text and chapters for every format, and the guards."""
import zipfile

import pytest

from summarizer import config, db, docparse, documents

from docs import make_docx, make_epub, make_pdf, make_scan

BIG = 10 ** 9


def parse(path, max_pages=BIG, max_chars=BIG):
    """Parses in-process (the sandbox wrapper has its own tests)."""
    return docparse.parse(path, max_pages, max_chars)


def code(path, **kw) -> str:
    """The DocError code parsing `path` raises."""
    with pytest.raises(docparse.DocError) as e:
        parse(path, **kw)
    return e.value.code


# ---------- format detection ----------

def test_formats_are_told_by_their_bytes_not_their_names(tmp_path):
    assert docparse.sniff(make_pdf(tmp_path / "book.txt", ["x"])) == "pdf"
    assert docparse.sniff(make_epub(tmp_path / "a.pdf", [("One", "x")])) == "epub"
    assert docparse.sniff(make_docx(tmp_path / "a.epub", [("", "x")])) == "docx"
    (tmp_path / "a.bin").write_text("Čas je zlato.\nDrugi del.", encoding="utf-8")
    assert docparse.sniff(tmp_path / "a.bin") == "txt"


@pytest.mark.parametrize("data", [
    b"\x89PNG\r\n\x1a\n" + b"\x00" * 100,                 # an image
    b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 100,  # old Word .doc
    b"BOOKMOBI" + b"\x00" * 100,                           # Kindle
])
def test_other_files_are_unsupported(tmp_path, data):
    (tmp_path / "f").write_bytes(data)
    assert code(tmp_path / "f") == "unsupported"


def test_a_plain_zip_is_unsupported(tmp_path):
    with zipfile.ZipFile(tmp_path / "a.epub", "w") as z:
        z.writestr("photo.jpg", b"x")
    assert code(tmp_path / "a.epub") == "unsupported"


# ---------- PDF ----------

def test_pdf_chapters_from_bookmarks(tmp_path):
    pdf = make_pdf(tmp_path / "a.pdf", ["Intro text", "more", "Second part", "end"],
                   [("Intro", 0), ("Second", 2)], title="Kniga", author="Ana")
    r = parse(pdf)
    assert (r["format"], r["title"], r["author"]) == ("pdf", "Kniga", "Ana")
    assert r["chapters"] == [{"title": "Intro", "start": 0, "end": 2}, {"title": "Second", "start": 2, "end": 4}]
    assert len(r["pages"]) == 4 and r["letters"][0] > 0


def test_pdf_chapters_from_headings_without_bookmarks(tmp_path):
    pages = ["Chapter 1\nThe start " * 3, "text", "Chapter 2 The middle\nmore", "text"]
    r = parse(make_pdf(tmp_path / "a.pdf", pages))
    assert [(c["title"], c["start"]) for c in r["chapters"]] == [("Chapter 1", 0), ("Chapter 2 The middle", 2)]


def test_pdf_without_structure_is_split_into_parts(tmp_path, monkeypatch):
    monkeypatch.setattr(docparse, "GROUP_CHARS", 100)
    r = parse(make_pdf(tmp_path / "a.pdf", ["word " * 20] * 6))
    assert [c["title"] for c in r["chapters"]][0].startswith("Part 1 (pages 1–")
    assert r["chapters"][-1]["end"] == 6 and len(r["chapters"]) > 1


def test_scanned_pdf_has_no_letters(tmp_path):
    r = parse(make_scan(tmp_path / "s.pdf", ["Hello", "World"]))
    assert r["letters"] == [0, 0]


def test_password_protected_pdf(tmp_path):
    from pypdf import PdfReader, PdfWriter
    src = make_pdf(tmp_path / "a.pdf", ["secret"])
    w = PdfWriter(clone_from=PdfReader(src))
    w.encrypt("pw")
    w.write(tmp_path / "locked.pdf")
    assert code(tmp_path / "locked.pdf") == "encrypted"


def test_page_limit(tmp_path):
    assert code(make_pdf(tmp_path / "a.pdf", ["a", "b", "c"]), max_pages=2) == "too_many_pages"


# ---------- EPUB ----------

def test_epub_chapters_from_its_table_of_contents(tmp_path):
    r = parse(make_epub(tmp_path / "a.epub", [("One", "para a\npara b"), ("Two", "para c")], title="Knjiga"))
    assert r["title"] == "Knjiga" and r["author"] == "Ana"
    assert [c["title"] for c in r["chapters"]] == ["One", "Two"]
    assert "para a" in r["pages"][0] and "p{}" not in r["pages"][0]  # styles aren't text


def test_epub_without_toc_uses_headings(tmp_path):
    chapters = [("Chapter 1", "first"), ("Chapter 2", "second"), ("Chapter 3", "third")]
    r = parse(make_epub(tmp_path / "a.epub", chapters, nav=False))
    assert [c["title"] for c in r["chapters"]] == ["Chapter 1", "Chapter 2", "Chapter 3"]


def test_epub_drm_is_refused_but_font_obfuscation_is_not(tmp_path):
    drm = '<encryption><EncryptedData><CipherData><CipherReference URI="OEBPS/ch1.xhtml"/></CipherData></EncryptedData></encryption>'
    fonts = '<encryption><EncryptedData><CipherData><CipherReference URI="OEBPS/font.otf"/></CipherData></EncryptedData></encryption>'
    assert code(make_epub(tmp_path / "a.epub", [("One", "x")], encryption=drm)) == "drm"
    assert parse(make_epub(tmp_path / "b.epub", [("One", "x")], encryption=fonts))["format"] == "epub"


@pytest.mark.filterwarnings("ignore:Duplicate name")  # the bomb replaces a chapter under the same name
def test_zip_bomb_is_refused(tmp_path):
    path = make_epub(tmp_path / "a.epub", [("One", "x")])
    with zipfile.ZipFile(path, "a", zipfile.ZIP_DEFLATED) as z:
        z.writestr("OEBPS/ch1.xhtml", b"<p>" + b"a" * (50 * 1024 ** 2) + b"</p>")  # ~50 KB compressed
    assert code(path) == "zip_bomb"


def test_too_many_zip_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(docparse, "ZIP_MAX_ENTRIES", 5)
    path = make_epub(tmp_path / "a.epub", [(f"C{i}", "x") for i in range(6)])
    assert code(path) == "zip_bomb"


# ---------- DOCX and TXT ----------

def test_docx_chapters_from_heading_styles(tmp_path):
    r = parse(make_docx(tmp_path / "a.docx", [("Heading1", "Uvod"), ("", "tekst"), ("Heading1", "Drugo"), ("", "več")],
                        title="Naslov"))
    assert r["title"] == "Naslov" and [c["title"] for c in r["chapters"]] == ["Uvod", "Drugo"]
    assert "več" in r["pages"][1]


def test_docx_without_styles_uses_headings(tmp_path):
    paras = [("", "Poglavje 1"), ("", "a"), ("", "Poglavje 2"), ("", "b")]
    assert [c["title"] for c in parse(make_docx(tmp_path / "a.docx", paras))["chapters"]] == ["Poglavje 1", "Poglavje 2"]


def test_txt_numbered_chapters_and_central_european_encoding(tmp_path):
    text = "Naslov knjige\n\n1\nZačetek\nbesedilo\n\n2\nSredina\nbesedilo\n\n3\nKonec\nbesedilo\n\nstran 4\n"
    (tmp_path / "a.txt").write_bytes(text.encode("cp1250"))
    r = parse(tmp_path / "a.txt")
    assert [c["title"] for c in r["chapters"]] == ["1 Začetek", "2 Sredina", "3 Konec"]


def test_many_small_chapters_are_merged(tmp_path, monkeypatch):
    monkeypatch.setattr(docparse, "GROUP_CHARS", 30)
    paras = [p for i in range(1, 61) for p in (("Heading1", f"Ch {i}"), ("", "x" * 20))]
    chapters = parse(make_docx(tmp_path / "a.docx", paras))["chapters"]
    assert len(chapters) <= docparse.MAX_CHAPTERS and chapters[0]["title"] == "Ch 1 – Ch 2"
    assert chapters[-1]["end"] == 60


# ---------- in the sandbox ----------

def test_parsing_runs_in_the_sandbox_and_stores_the_text(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    pdf = make_pdf(job / "a.pdf", ["Chapter 1\nHello", "Chapter 2\nBye"])
    parsed = documents.parse(pdf, job)
    digest = documents.sha256(pdf)
    documents.store(digest, "a.pdf", parsed)
    doc = db.get_document(digest)
    assert doc["format"] == "pdf" and doc["pages"] == 2 and len(doc["chapters"]) == 2
    assert "Hello" in db.get_pages(digest)[0]


def test_sandbox_sees_no_repository_and_no_network(tmp_path):
    from summarizer import proc, sandbox
    probe = ("import os, socket\n"
             f"print(os.path.exists({str(config.ROOT / '.env')!r}), os.path.exists({str(config.ROOT / 'bot.py')!r}))\n"
             "try:\n socket.create_connection(('1.1.1.1', 53), timeout=2); print('net')\n"
             "except OSError: print('no net')\n"
             "print(sorted(os.environ))")
    out = proc.run(sandbox.command(tmp_path, ["python", "-c", probe]), timeout=60).stdout.splitlines()
    assert out[0] == "False False" and out[1] == "no net"
    assert "TELEGRAM_BOT_TOKEN" not in out[2] and "DATA_DIR" not in out[2]


def test_sandbox_errors_become_user_messages(tmp_path):
    (tmp_path / "x.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    with pytest.raises(documents.DocumentError, match="PDF, EPUB, DOCX and TXT"):
        documents.parse(tmp_path / "x.png", tmp_path)


def test_unfinished_documents_are_marked_at_startup():
    db.save_document("abc", name="x", status="processing")
    db.fail_stale_requests()
    assert db.get_document("abc")["status"] == "failed"
