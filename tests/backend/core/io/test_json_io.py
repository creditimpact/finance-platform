from __future__ import annotations

import json
from pathlib import Path

from backend.core.io.json_io import update_json_in_place


def test_update_json_in_place_creates_file(tmp_path: Path) -> None:
    target = tmp_path / "data.json"

    def _apply(payload: object) -> dict[str, object]:
        data = dict(payload) if isinstance(payload, dict) else {}
        data["value"] = 3
        return data

    result = update_json_in_place(target, _apply)

    assert target.exists()
    assert json.loads(target.read_text(encoding="utf-8")) == {"value": 3}
    assert result == {"value": 3}


def test_update_json_in_place_skips_when_unchanged(tmp_path: Path) -> None:
    target = tmp_path / "data.json"
    target.write_text(json.dumps({"value": 1}, ensure_ascii=False, indent=2), encoding="utf-8")

    before_stat = target.stat()

    def _noop(payload: object) -> dict[str, object]:
        if isinstance(payload, dict):
            return dict(payload)
        return {}

    result = update_json_in_place(target, _noop)

    after_stat = target.stat()

    assert json.loads(target.read_text(encoding="utf-8")) == {"value": 1}
    assert result == {"value": 1}
    assert before_stat.st_mtime_ns == after_stat.st_mtime_ns
