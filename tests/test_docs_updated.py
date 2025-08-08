import subprocess
from pathlib import Path


def _changed_files() -> set[Path]:
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "HEAD^", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError:
        return set()
    return {Path(p) for p in result.stdout.splitlines() if p}


def test_model_docs_updated():
    changed = _changed_files()
    model_changed = any(p.parts and p.parts[0] == "models" and p.suffix == ".py" for p in changed)
    if model_changed:
        assert Path("docs/DATA_MODELS.md") in changed, (
            "models/ changed without updating docs/DATA_MODELS.md"
        )


def test_system_overview_updated():
    changed = _changed_files()
    if Path("orchestrators.py") in changed:
        diff = subprocess.run(
            ["git", "diff", "HEAD^", "HEAD", "--", "orchestrators.py"],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout
        api_changed = any(
            line.startswith("+def ") or line.startswith("-def ") for line in diff.splitlines()
        )
        if api_changed:
            assert Path("docs/SYSTEM_OVERVIEW.md") in changed, (
                "orchestrators.py public API changed without updating docs/SYSTEM_OVERVIEW.md"
            )
