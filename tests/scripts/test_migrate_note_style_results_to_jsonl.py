from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.migrate_note_style_results_to_jsonl import (
    LEGACY_FAILURE_PAYLOAD,
    migrate_file,
    migrate_legacy_note_style_results,
)


def _write_result(tmp_path: Path, relative: str, content: str) -> Path:
    file_path = tmp_path / relative
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    return file_path


def test_migrate_file_creates_jsonl_from_json_line(tmp_path: Path) -> None:
    result_path = _write_result(
        tmp_path,
        "runs/sid123/ai_packs/note_style/results/sample.result",
        '{"status":"ok"}\n',
    )

    jsonl_path = migrate_file(result_path)

    assert jsonl_path is not None
    data = jsonl_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in data] == [{"status": "ok"}]


def test_migrate_file_wraps_non_json_lines(tmp_path: Path) -> None:
    result_path = _write_result(
        tmp_path,
        "runs/sid456/ai_packs/note_style/results/legacy.result",
        "not-json\n",
    )

    jsonl_path = migrate_file(result_path)

    assert jsonl_path is not None
    content = jsonl_path.read_text(encoding="utf-8").strip()
    payload = json.loads(content)
    expected = dict(LEGACY_FAILURE_PAYLOAD)
    expected["raw"] = "not-json"
    assert payload == expected


def test_migrate_file_skips_when_jsonl_exists(tmp_path: Path) -> None:
    result_path = _write_result(
        tmp_path,
        "runs/sid789/ai_packs/note_style/results/existing.result",
        '{"status":"ok"}\n',
    )
    existing_jsonl = result_path.with_suffix(".jsonl")
    existing_jsonl.write_text("", encoding="utf-8")

    created = migrate_file(result_path)
    assert created is None


def test_migrate_legacy_note_style_results_processes_all(tmp_path: Path) -> None:
    first = _write_result(
        tmp_path,
        "runs/one/ai_packs/note_style/results/a.result",
        '{"first": 1}\n',
    )
    second = _write_result(
        tmp_path,
        "runs/two/ai_packs/note_style/results/b.result",
        "bad\n",
    )

    migrated = migrate_legacy_note_style_results(tmp_path)

    assert {path.name for path in migrated} == {"a.jsonl", "b.jsonl"}
    assert json.loads(first.with_suffix(".jsonl").read_text(encoding="utf-8")) == {
        "first": 1
    }
    wrapped = json.loads(second.with_suffix(".jsonl").read_text(encoding="utf-8"))
    assert wrapped["status"] == LEGACY_FAILURE_PAYLOAD["status"]
    assert wrapped["error"] == LEGACY_FAILURE_PAYLOAD["error"]
    assert wrapped["raw"] == "bad"

