import ast
import re
from pathlib import Path


def test_no_raw_template_paths():
    root = Path(__file__).resolve().parents[1]
    pattern = re.compile(r"_letter_template\.html$")
    allowed = {
        root / "backend" / "core" / "letters" / "router.py",
        root / "backend" / "core" / "letters" / "validators.py",
    }
    for path in root.rglob("*.py"):
        if path.is_relative_to(root / "tests"):
            continue
        if path in allowed:
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if pattern.search(node.value):
                    raise AssertionError(f"Found template literal {node.value!r} in {path}")
