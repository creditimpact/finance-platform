import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module)
    return imports


def test_letter_rendering_does_not_import_gpt_prompting() -> None:
    path = PROJECT_ROOT / "logic" / "letter_rendering.py"
    imports = _imports(path)
    assert "logic.gpt_prompting" not in imports and "gpt_prompting" not in imports


def test_compliance_modules_are_pure() -> None:
    for module_path in (PROJECT_ROOT / "logic").glob("compliance_*.py"):
        imports = _imports(module_path)
        for name in imports:
            assert (
                "rendering" not in name and "prompt" not in name
            ), f"{module_path} imports {name}"


def test_features_do_not_import_app() -> None:
    features_dir = PROJECT_ROOT / "features"
    if not features_dir.exists():
        return
    for file_path in features_dir.rglob("*.py"):
        imports = _imports(file_path)
        assert "app" not in imports, f"{file_path} imports app.py"


def test_no_module_imports_main() -> None:
    """Ensure core modules do not depend on the CLI layer."""
    for file_path in PROJECT_ROOT.rglob("*.py"):
        if file_path.name == "main.py" or "tests" in file_path.parts:
            continue
        imports = _imports(file_path)
        assert "main" not in imports and not any(
            name.startswith("main.") for name in imports
        ), f"{file_path} imports main"
