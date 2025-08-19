import ast
from pathlib import Path


def test_instruction_no_long_strings():
    root = Path(__file__).resolve().parents[1]
    targets = [
        root / "backend" / "core" / "logic" / "rendering" / "instruction_renderer.py",
        root / "backend" / "core" / "logic" / "rendering" / "instructions_generator.py",
    ]
    for path in targets:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if getattr(node, "lineno", None) == 1:
                    continue
                if len(node.value.strip()) > 80:
                    raise AssertionError(
                        f"Found long string literal in {path}: {node.value[:40]!r}"
                    )
