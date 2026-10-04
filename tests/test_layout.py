"""AGENTS.md rules for the code layout: how the tgbot package imports, and what summarizer may import."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TGBOT = sorted((ROOT / "tgbot").glob("*.py"))
MODULES = {p.stem for p in TGBOT} - {"__init__"}


def _imports(path: Path):
    """The import statements of a file."""
    return [n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, (ast.Import, ast.ImportFrom))]


def test_tgbot_imports_modules_never_names():
    # A name imported with `from x import y` keeps the original when a test (or reset) rebinds x.y.
    bad = []
    for path in TGBOT + [ROOT / "bot.py"] + sorted((ROOT / "tests").glob("*.py")):
        for node in _imports(path):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("tgbot."):
                bad.append(f"{path.name}: from {node.module} import ...")
            elif isinstance(node, ast.ImportFrom) and node.module == "tgbot":
                bad += [f"{path.name}: {a.name}" for a in node.names if a.name not in MODULES]
    assert not bad


def test_summarizer_never_imports_the_bot_side():
    # summarizer/ is mounted into the sandboxes and runs there without Telegram or the bot.
    bad = []
    for path in sorted((ROOT / "summarizer").glob("*.py")):
        for node in _imports(path):
            names = [node.module or ""] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
            bad += [f"{path.name}: {n}" for n in names if n.split(".")[0] in ("tgbot", "telegram", "bot", "access")]
    assert not bad


def test_thread_pools_come_from_proc_pool():
    # A plain ThreadPoolExecutor's threads don't get the job's cancel event: a cancel wouldn't reach them.
    bad = [p.name for p in sorted((ROOT / "summarizer").glob("*.py")) + TGBOT + [ROOT / "bot.py"]
           if p.name != "proc.py" and "ThreadPoolExecutor(" in p.read_text()]
    assert not bad
