from __future__ import annotations

import json
from pathlib import Path

from backend.core.ai.paths import ensure_note_style_account_paths, ensure_note_style_paths
from scripts import rebuild_note_style_packs as module


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_rebuild_note_style_builds_from_responses(tmp_path: Path) -> None:
    sid = "SID200"
    account_id = "idx-001"

    run_dir = tmp_path / sid
    response_dir = run_dir / "frontend" / "review" / "responses"
    account_dir = run_dir / "cases" / "accounts" / account_id

    _write_json(
        response_dir / f"{account_id}.result.json",
        {"answers": {"explain": "The bank reported a mistake, please fix."}},
    )

    _write_json(account_dir / "summary.json", {"account_id": account_id})
    _write_json(account_dir / "meta.json", {"account_id": account_id})
    _write_json(account_dir / "bureaus.json", {"experian": {"reported_creditor": "Bank"}})
    _write_json(account_dir / "tags.json", [])

    module._process_sid(sid, tmp_path)

    paths = ensure_note_style_paths(tmp_path, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)

    assert account_paths.pack_file.exists()
    assert not account_paths.result_file.exists()

    index_payload = json.loads(paths.index_file.read_text(encoding="utf-8"))
    assert index_payload["packs"][0]["status"] == "built"

    runflow_payload = json.loads((run_dir / "runflow.json").read_text(encoding="utf-8"))
    stage_payload = runflow_payload["stages"]["note_style"]
    assert stage_payload["status"] == "built"
    assert stage_payload["metrics"]["packs_total"] == 1


def test_rebuild_note_style_marks_empty_success(tmp_path: Path) -> None:
    sid = "SID201"

    module._process_sid(sid, tmp_path)

    run_dir = tmp_path / sid
    runflow_payload = json.loads((run_dir / "runflow.json").read_text(encoding="utf-8"))
    stage_payload = runflow_payload["stages"]["note_style"]

    assert stage_payload["status"] == "success"
    assert stage_payload["empty_ok"] is True
    assert stage_payload["metrics"]["packs_total"] == 0
