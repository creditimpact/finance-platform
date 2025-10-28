from pathlib import Path

from backend.pipeline.runs import safe_replace


def test_safe_replace_overwrites_with_latest_content(tmp_path: Path) -> None:
    target = tmp_path / "sample.json"

    safe_replace(str(target), "first write")
    assert target.read_text(encoding="utf-8") == "first write"

    safe_replace(str(target), "second write")
    assert target.read_text(encoding="utf-8") == "second write"
