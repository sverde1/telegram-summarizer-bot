"""Text recognition (OCR) for scanned PDFs: page rendering, the two engines, language check, estimates.

Pages are rendered with poppler and read by Tesseract (default) or RapidOCR, every step in the sandbox (the
PDF is untrusted, and so is what the engines parse). On the CPU several processes run side by side
(OCR_WORKERS), one thread each; on a GPU (RapidOCR with OCR_DEVICE=cuda) one process does it all. Pages go in
batches; finished pages are stored right away, so a cancelled or interrupted run
continues where it stopped.

Languages: Tesseract needs a trained model per language (`data/tessdata`, seeded with the system's English
and script-detection files); RapidOCR needs a recognition model per script. Which languages are installed is
a setting (`ocr_languages`, managed with /ocrlang). Before OCR is offered, a few sample pages are read to
check the language: a scan in a language that isn't installed is refused, with the list of those that are.
"""
import json
import logging
import shutil
import statistics
from collections.abc import Callable
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from pathlib import Path

from . import config, db, fetch, proc, sandbox, stats

log = logging.getLogger(__name__)

BATCH = 10            # pages per OCR process run (a batch is the cancel granularity, ~10-30 s)
RENDER_EDGE = 2500    # pixels on the longer side: plenty for OCR, and bounds memory whatever the page size
SAMPLE_PAGES = 3
LOW_CONFIDENCE = 65   # Tesseract mean word confidence (0-100) below which a language mismatch is believed
TIMEOUT = 900         # seconds per batch
# Seconds per page with all workers busy, until measured on this machine.
SPEED_DEFAULT = {"tesseract": 1.5, "rapidocr": 4.0, "rapidocr-cuda": 0.3}
TESSDATA_SYSTEM = sorted(Path("/usr/share/tesseract-ocr").glob("*/tessdata"))

# code (Tesseract's) -> (name, ISO 639-1 as py3langid reports it, script as Tesseract's OSD names it,
# RapidOCR recognition model: "" = the default one (Chinese + English), None = not available)
LANGUAGES = {
    "eng": ("English", "en", "Latin", ""), "slv": ("Slovenian", "sl", "Latin", "latin"),
    "hrv": ("Croatian", "hr", "Latin", "latin"), "bos": ("Bosnian", "bs", "Latin", "latin"),
    "srp_latn": ("Serbian (Latin)", "sr", "Latin", "latin"), "srp": ("Serbian (Cyrillic)", "sr", "Cyrillic", "cyrillic"),
    "mkd": ("Macedonian", "mk", "Cyrillic", "cyrillic"), "bul": ("Bulgarian", "bg", "Cyrillic", "cyrillic"),
    "rus": ("Russian", "ru", "Cyrillic", "cyrillic"), "ukr": ("Ukrainian", "uk", "Cyrillic", "cyrillic"),
    "bel": ("Belarusian", "be", "Cyrillic", "cyrillic"), "deu": ("German", "de", "Latin", "latin"),
    "fra": ("French", "fr", "Latin", "latin"), "ita": ("Italian", "it", "Latin", "latin"),
    "spa": ("Spanish", "es", "Latin", "latin"), "por": ("Portuguese", "pt", "Latin", "latin"),
    "nld": ("Dutch", "nl", "Latin", "latin"), "pol": ("Polish", "pl", "Latin", "latin"),
    "ces": ("Czech", "cs", "Latin", "latin"), "slk": ("Slovak", "sk", "Latin", "latin"),
    "hun": ("Hungarian", "hu", "Latin", "latin"), "ron": ("Romanian", "ro", "Latin", "latin"),
    "swe": ("Swedish", "sv", "Latin", "latin"), "dan": ("Danish", "da", "Latin", "latin"),
    "nor": ("Norwegian", "no", "Latin", "latin"), "fin": ("Finnish", "fi", "Latin", "latin"),
    "est": ("Estonian", "et", "Latin", "latin"), "lav": ("Latvian", "lv", "Latin", "latin"),
    "lit": ("Lithuanian", "lt", "Latin", "latin"), "tur": ("Turkish", "tr", "Latin", "latin"),
    "cat": ("Catalan", "ca", "Latin", "latin"), "sqi": ("Albanian", "sq", "Latin", "latin"),
    "ell": ("Greek", "el", "Greek", None), "ara": ("Arabic", "ar", "Arabic", "arabic"),
    "heb": ("Hebrew", "he", "Hebrew", None), "hin": ("Hindi", "hi", "Devanagari", "devanagari"),
    "chi_sim": ("Chinese (simplified)", "zh", "Han", ""), "chi_tra": ("Chinese (traditional)", "zh", "Han", "chinese_cht"),
    "jpn": ("Japanese", "ja", "Japanese", "japan"), "kor": ("Korean", "ko", "Hangul", "korean"),
}
# OSD names that mean the same script as the table's.
_SCRIPT_ALIASES = {"Katakana": "Japanese", "Hiragana": "Japanese", "Korean": "Hangul", "HanS": "Han", "HanT": "Han"}

NO_ENGINE = "⚠️ Text recognition (OCR) isn't set up on this bot, so I can't read scanned documents yet."


class UnsupportedLanguage(Exception):
    """A scan in a language that isn't installed. str() is the user message; `code` the language, if known."""

    def __init__(self, message: str, code: str | None = None):
        """Stores the user message and the detected language's code."""
        super().__init__(message)
        self.code = code


def engine() -> str:
    """The configured OCR engine."""
    return config.OCR_ENGINE if config.OCR_ENGINE in ("tesseract", "rapidocr") else "rapidocr"


_cuda_ok: bool | None = None  # whether onnxruntime can use CUDA here (checked once)


def device() -> str:
    """Where OCR runs: "cuda" when asked for, RapidOCR is the engine and CUDA works; else "cpu".

    Like Whisper, a missing GPU or CUDA runtime means the CPU, with a warning, not a broken OCR.
    """
    global _cuda_ok
    if config.OCR_DEVICE != "cuda" or engine() != "rapidocr":
        return "cpu"
    if _cuda_ok is None:
        try:
            import onnxruntime
            _cuda_ok = "CUDAExecutionProvider" in onnxruntime.get_available_providers()
        except ImportError:
            _cuda_ok = False
        if not _cuda_ok:
            log.warning("OCR_DEVICE=cuda, but onnxruntime has no CUDA support (install onnxruntime-gpu); "
                        "OCR runs on the CPU")
    return "cuda" if _cuda_ok else "cpu"


def _speed_key() -> str:
    """Stats key for the OCR speed of the engine and device in use (a GPU is much faster)."""
    return engine() + ("-cuda" if device() == "cuda" else "")


def available() -> bool:
    """Whether the configured engine can run (Tesseract must be installed as a system package)."""
    if engine() == "tesseract":
        return shutil.which("tesseract") is not None
    try:
        import rapidocr  # noqa: F401
        return True
    except ImportError:
        return False


def _usable(code: str) -> bool:
    """Whether the current engine has what it needs to read this language."""
    if engine() == "tesseract":
        return (tessdata() / f"{code}.traineddata").exists()
    model = LANGUAGES[code][3]
    return model == "" or (model is not None and (rapid_models() / f"{model}.onnx").exists())


def installed() -> list[str]:
    """The installed OCR languages (codes) the current engine can read; English by default.

    A language added for one engine isn't usable by the other until added again (switching OCR_ENGINE).
    """
    stored = db.get_setting("ocr_languages")
    codes = [c for c in (stored or "eng").split(",") if c in LANGUAGES and _usable(c)]
    return codes or ["eng"]


def names(codes: list[str]) -> str:
    """Human-readable list, e.g. "English, Slovenian"."""
    return ", ".join(LANGUAGES[c][0] for c in codes if c in LANGUAGES)


def tessdata() -> Path:
    """The bot's Tesseract model folder, seeded from the system's English and script-detection models."""
    folder = config.DATA_DIR / "tessdata"
    folder.mkdir(exist_ok=True)
    for name in ("eng.traineddata", "osd.traineddata"):
        if not (folder / name).exists():
            for system in TESSDATA_SYSTEM:
                if (system / name).exists():
                    shutil.copy(system / name, folder / name)
                    break
    return folder


def rapid_models() -> Path:
    """Folder of RapidOCR recognition models installed by /ocrlang."""
    folder = config.DATA_DIR / "rapidocr"
    folder.mkdir(exist_ok=True)
    return folder


def seconds_per_page() -> float:
    """Measured wall-clock seconds per page with all workers busy (a starting guess until measured)."""
    return stats.get(f"ocr:{_speed_key()}", SPEED_DEFAULT[_speed_key()])


def estimate(pages: int) -> float:
    """Estimated seconds to OCR this many pages."""
    return pages * seconds_per_page()


# ---------- running the engines ----------

def _render(pdf: Path, pages: list[int], workdir: Path, tag: str) -> dict[int, Path]:
    """Renders PDF pages (0-based) to grayscale PNGs in the sandbox.

    Returns:
        {page: image path}.
    """
    out = workdir / f"img-{tag}"
    out.mkdir(exist_ok=True)
    for page in pages:
        args = ["pdftoppm", "-gray", "-png", "-singlefile", "-scale-to", str(RENDER_EDGE),
                "-f", str(page + 1), "-l", str(page + 1), f"/job/{pdf.relative_to(workdir)}",
                f"/job/{out.relative_to(workdir)}/p{page}"]
        proc.run(sandbox.command(workdir, args, memory=2 * 1024 ** 3), timeout=TIMEOUT)
    return {p: out / f"p{p}.png" for p in pages if (out / f"p{p}.png").exists()}


def _tesseract(images: list[Path], langs: str, workdir: Path, tag: str, tsv: bool = False) -> list[str]:
    """Runs Tesseract over images in one process (a list file), returning each page's text (or TSV)."""
    listing = workdir / f"list-{tag}.txt"
    listing.write_text("".join(f"/job/{p.relative_to(workdir)}\n" for p in images))
    args = ["tesseract", f"/job/{listing.name}", "-", "-l", langs, "-c", "page_separator=\f"]
    if tsv:
        args.append("tsv")
    run = proc.run(sandbox.command(workdir, args, ro_binds={tessdata(): "/tessdata"},
                                   env={"TESSDATA_PREFIX": "/tessdata"}, memory=3 * 1024 ** 3), timeout=TIMEOUT)
    if run.returncode != 0:
        raise RuntimeError(f"tesseract exited {run.returncode}: {run.stderr[-300:]}")
    if tsv:  # one TSV table for all pages; split them by the page_num column
        pages: dict[int, list[str]] = {}
        for line in run.stdout.splitlines()[1:]:
            cols = line.split("\t")
            if len(cols) == 12:
                pages.setdefault(int(cols[1]), []).append(line)
        return ["\n".join(pages.get(i + 1, [])) for i in range(len(images))]
    texts = run.stdout.split("\f")
    return (texts + [""] * len(images))[:len(images)]


def _rapidocr(images: list[Path], langs: str, workdir: Path, tag: str) -> list[tuple[str, float]]:
    """Runs RapidOCR over images in one sandboxed process; returns (text, mean score) per page."""
    model = LANGUAGES.get(langs.split("+")[0], ("", "", "", ""))[3]
    rec, binds = "-", {}
    if model:
        path = rapid_models() / f"{model}.onnx"
        if not path.exists():
            raise RuntimeError(f"RapidOCR model {model} isn't installed")
        rec, binds = f"/models/{path.name}", {rapid_models(): "/models"}
    out = workdir / f"rapid-{tag}.json"
    args = ["python", "-m", "summarizer.rapid_ocr", f"/job/{out.name}", rec, device(),
            *[f"/job/{p.relative_to(workdir)}" for p in images]]
    run = proc.run(sandbox.command(workdir, args, ro_binds=binds, gpu=device() == "cuda"), timeout=TIMEOUT)
    if not out.exists():
        raise RuntimeError(f"rapidocr exited {run.returncode}: {run.stderr[-300:]}")
    return [(r["text"], r["score"]) for r in json.loads(out.read_text())]


def _read_batch(pdf: Path, pages: list[int], langs: str, workdir: Path, tag: str) -> dict[int, str]:
    """Renders and recognises one batch of pages; returns {page: text}."""
    images = _render(pdf, pages, workdir, tag)
    order = [p for p in pages if p in images]
    if engine() == "tesseract":
        texts = _tesseract([images[p] for p in order], langs, workdir, tag)
    else:
        texts = [t for t, _ in _rapidocr([images[p] for p in order], langs, workdir, tag)]
    for path in images.values():
        path.unlink(missing_ok=True)  # don't keep a whole book of page images around
    return {p: text.strip() for p, text in zip(order, texts)} | {p: "" for p in pages if p not in images}


def run(pdf: Path, sha256: str, pages: list[int], langs: str, workdir: Path,
        status: Callable[[str, float | None], None]) -> None:
    """OCRs the given pages in parallel batches, storing each batch's text as soon as it's done.

    Args:
        pdf: The scanned PDF, inside workdir.
        sha256: The document whose pages are stored.
        pages: 0-based page numbers to read.
        langs: Tesseract language string, e.g. "slv" or "eng+slv".
        workdir: The job's directory.
        status: Callback (text, eta); also the cancel checkpoint between batches.

    Raises:
        proc.ProcCancelled: The job was cancelled (finished batches stay stored).
    """
    import time
    batches = [pages[i:i + BATCH] for i in range(0, len(pages), BATCH)]
    done, started = 0, time.monotonic()
    status(f"🔍 Recognizing text (OCR): page 1 of {len(pages)}…", estimate(len(pages)))
    workers = 1 if device() == "cuda" else max(1, config.OCR_WORKERS)  # a GPU is fed by one process
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(_read_batch, pdf, batch, langs, workdir, str(k)): batch
                   for k, batch in enumerate(batches)}
        try:
            while pending:
                finished, _ = wait(pending, return_when=FIRST_EXCEPTION)
                for future in finished:
                    batch = pending.pop(future)
                    texts = future.result()  # re-raises a batch's error (or cancel) here
                    db.save_pages(sha256, texts, f"ocr-{engine()}")
                    done += len(batch)
                    left = len(pages) - done
                    per_page = (time.monotonic() - started) / done
                    if left:
                        status(f"🔍 Recognizing text (OCR): page {done + 1} of {len(pages)}…", left * per_page)
        except BaseException:
            for future in pending:
                future.cancel()
            proc.current_job_cancel.set()  # stop the batches still running (their programs get killed)
            raise
    if len(pages) >= 5:  # tiny runs are dominated by start-up time
        stats.record(f"ocr:{_speed_key()}", (time.monotonic() - started) / len(pages))


# ---------- language check ----------

def _detect(text: str) -> str | None:
    """ISO 639-1 language of a text (py3langid), or None when there's too little text to tell."""
    if sum(c.isalpha() for c in text) < 100:
        return None
    import py3langid
    return py3langid.classify(text)[0]


def _script(image: Path, workdir: Path) -> str | None:
    """The script Tesseract's orientation/script detection sees on a page (e.g. "Latin"), or None."""
    run = proc.run(sandbox.command(workdir, ["tesseract", f"/job/{image.relative_to(workdir)}", "-", "--psm", "0"],
                                   ro_binds={tessdata(): "/tessdata"}, env={"TESSDATA_PREFIX": "/tessdata"}),
                   timeout=120)
    for line in (run.stdout + run.stderr).splitlines():
        if line.startswith("Script:"):
            script = line.split(":", 1)[1].strip()
            return _SCRIPT_ALIASES.get(script, script)
    return None


def choose_languages(pdf: Path, pages: list[int], workdir: Path) -> str:
    """Reads a few sample pages to pick the OCR language(s), or refuses a language that isn't installed.

    The detected language decides only when it's clear: py3langid confuses close languages (Slovenian,
    Croatian, Serbian…) on OCR text, so a scan is refused only if the detected language isn't installed
    AND the recognition confidence with the installed ones is low (a real mismatch reads badly).

    Args:
        pdf: The scanned PDF, inside workdir.
        pages: 0-based pages that need OCR (samples are taken from their middle, past any front matter).

    Returns:
        The language string for Tesseract, e.g. "slv" or "eng+slv".

    Raises:
        UnsupportedLanguage: The scan's script or language isn't installed (message lists what is).
    """
    have = installed()
    supported = f"Supported: {names(have)}."
    mid = len(pages) // 2
    sample = pages[max(0, mid - 1):mid - 1 + SAMPLE_PAGES] or pages[:SAMPLE_PAGES]
    images = _render(pdf, sample, workdir, "sample")
    if not images:
        return "+".join(have)
    first = next(iter(images.values()))
    if available_tesseract():  # script detection needs Tesseract, whichever engine reads the pages
        script = _script(first, workdir)
        scripts = {LANGUAGES[c][2] for c in have}
        if script and script not in scripts and script in {v[2] for v in LANGUAGES.values()}:
            raise UnsupportedLanguage(f"⚠️ This scan is in {script} script, which text recognition doesn't "
                                      f"support yet. {supported}")
    if engine() == "tesseract":
        tables = _tesseract(list(images.values()), "+".join(have), workdir, "sample", tsv=True)
        words, confs = [], []
        for table in tables:
            for row in table.splitlines():
                cols = row.split("\t")
                if cols[11].strip() and float(cols[10]) >= 0:
                    words.append(cols[11])
                    confs.append(float(cols[10]))
        text, confidence = " ".join(words), (statistics.mean(confs) if confs else 0)
    else:
        results = _rapidocr(list(images.values()), "+".join(have), workdir, "sample")
        text = "\n".join(t for t, _ in results)
        confidence = 100 * statistics.mean([s for _, s in results]) if results else 0
    for path in images.values():
        path.unlink(missing_ok=True)
    iso = _detect(text)
    detected = [c for c, v in LANGUAGES.items() if v[1] == iso]
    log.info("OCR sample: language %s, confidence %.0f, installed %s", iso, confidence, have)
    if usable := [c for c in detected if c in have]:
        return "+".join(usable)
    if detected and confidence < LOW_CONFIDENCE:
        raise UnsupportedLanguage(f"⚠️ This scan looks like {LANGUAGES[detected[0]][0]}, which text recognition "
                                  f"doesn't support yet. {supported}", detected[0])
    return "+".join(have)


def available_tesseract() -> bool:
    """Whether Tesseract is installed (also used for script detection when RapidOCR reads the pages)."""
    return shutil.which("tesseract") is not None


# ---------- adding languages (/ocrlang) ----------

TESSDATA_URL = "https://github.com/tesseract-ocr/tessdata_fast/raw/main/{code}.traineddata"
# Downloads go only to these hosts over HTTPS, redirects included (GitHub serves raw files from a CDN host).
DOWNLOAD_HOSTS = {"github.com", "raw.githubusercontent.com", "objects.githubusercontent.com",
                  "www.modelscope.cn", "modelscope.cn"}
MAX_DOWNLOAD = 60 * 1024 ** 2  # the biggest model in question is ~25 MB


class LanguageError(Exception):
    """Adding or removing a language failed; str() is the message for the admin."""


def _download(url: str, dest: Path, sha256: str | None = None) -> None:
    """Downloads a model file from the allowed hosts (see fetch.download).

    Raises:
        LanguageError: The download failed or the file isn't what was expected.
    """
    def allowed(host: str) -> bool:
        """GitHub's hosts, and ModelScope with its regional CDN hosts (the SHA-256 checks the content)."""
        return host in DOWNLOAD_HOSTS or host.endswith(".modelscope.cn")

    try:
        fetch.download(url, dest, allowed=allowed, max_bytes=MAX_DOWNLOAD, sha256=sha256)
    except fetch.FetchError as e:
        raise LanguageError({"too_large": "the file is too large",
                             "checksum": "the downloaded file doesn't match its checksum",
                             "blocked": f"download redirected to an unexpected place ({e.detail})"}.get(
                                 e.code, f"download failed: {e.detail}"))


def _rapid_model_source(model: str) -> tuple[str, str]:
    """URL and SHA-256 of a RapidOCR recognition model, from the table that ships with RapidOCR itself."""
    import rapidocr
    import yaml
    table = yaml.safe_load((Path(rapidocr.__file__).parent / "default_models.yaml").read_text())
    for version in table["onnxruntime"].values():
        for name, info in (version.get("rec") or {}).items():
            if name.startswith(f"{model}_PP-OCR") and name.endswith("_mobile"):
                return info["model_dir"], info["SHA256"]
    raise LanguageError(f"RapidOCR has no model for {model}")


def _check_tesseract_model(path: Path, code: str) -> None:
    """Runs Tesseract with a new language model on a tiny image, in the sandbox, to prove the file works.

    Raises:
        LanguageError: Tesseract can't use it.
    """
    from PIL import Image, ImageDraw
    work = config.DATA_DIR / "work" / f"ocrlang-{code}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        img = Image.new("L", (400, 80), 255)
        ImageDraw.Draw(img).text((10, 30), "Test 123", fill=0)
        img.save(work / "t.png")
        models = work / "tessdata"
        models.mkdir()
        shutil.copy(path, models / f"{code}.traineddata")
        run = proc.run(sandbox.command(work, ["tesseract", "/job/t.png", "-", "-l", code],
                                       ro_binds={models: "/tessdata"}, env={"TESSDATA_PREFIX": "/tessdata"}),
                       timeout=120)
        if run.returncode != 0:
            raise LanguageError(f"Tesseract can't use the downloaded model: {run.stderr[-200:]}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def add_language(code: str) -> str:
    """Installs an OCR language for the current engine and adds it to the installed list.

    Returns:
        A confirmation for the admin.

    Raises:
        LanguageError: Unknown code, not available for the engine, or the download failed.
    """
    if code not in LANGUAGES:
        raise LanguageError(f"Unknown language code {code!r}. Send /ocrlang for the list.")
    name, _, _, model = LANGUAGES[code]
    if engine() == "tesseract":
        target = tessdata() / f"{code}.traineddata"
        if not target.exists():
            tmp = tessdata() / f"{code}.download"
            try:
                _download(TESSDATA_URL.format(code=code), tmp)
                _check_tesseract_model(tmp, code)
                tmp.replace(target)
            finally:
                tmp.unlink(missing_ok=True)
    elif model is None:
        raise LanguageError(f"RapidOCR can't read {name}. Use OCR_ENGINE=tesseract for it.")
    elif model and not (rapid_models() / f"{model}.onnx").exists():
        url, sha = _rapid_model_source(model)
        _download(url, rapid_models() / f"{model}.onnx", sha)
    codes = [c for c in (db.get_setting("ocr_languages") or "eng").split(",") if c in LANGUAGES]
    if code not in codes:
        codes.append(code)
    db.set_setting("ocr_languages", ",".join(codes))
    return f"✅ {name} ({code}) added. Text recognition now reads: {names(installed())}."


def remove_language(code: str) -> str:
    """Removes an OCR language from the installed list (the last one can't be removed).

    Raises:
        LanguageError: Not installed, or the last language.
    """
    codes = [c for c in (db.get_setting("ocr_languages") or "eng").split(",") if c in LANGUAGES]
    if code not in codes:
        raise LanguageError(f"{code!r} isn't installed.")
    if len(codes) == 1:
        raise LanguageError("That's the only language; add another one first.")
    codes.remove(code)
    db.set_setting("ocr_languages", ",".join(codes))
    if code not in ("eng", "osd"):
        (tessdata() / f"{code}.traineddata").unlink(missing_ok=True)
    return f"✅ {LANGUAGES[code][0]} ({code}) removed. Text recognition reads: {names(installed())}."
