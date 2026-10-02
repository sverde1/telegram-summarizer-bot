"""AGENTS.md rule: every function and method has a docstring (test functions themselves are exempt)."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = [ROOT / "bot.py", ROOT / "access.py", *sorted((ROOT / "summarizer").glob("*.py")),
         *sorted((ROOT / "tgbot").rglob("*.py")), *sorted((ROOT / "tests").glob("*.py"))]


def test_every_function_has_a_docstring():
    missing = []
    for path in FILES:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not ast.get_docstring(node):
                if not node.name.startswith("test_"):
                    missing.append(f"{path.relative_to(ROOT)}:{node.lineno} {node.name}")
    assert not missing, "functions without a docstring:\n" + "\n".join(missing)
