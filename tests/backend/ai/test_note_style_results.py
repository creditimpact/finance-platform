from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.ai.note_style_results import store_note_style_result
from backend.ai.note_style_stage import build_note_style_pack_for_account
from backend.core.ai.paths import ensure_note_style_account_paths, ensure_note_style_paths


def _write_response(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def test_store_note_style_result_updates_index_and_triggers_refresh(
    tmp_path: Path, monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    sid = "SID900"
    account_id = "idx-900"
    runs_root = tmp_path / "runs"
    response_dir = runs_root / sid / "frontend" / "review" / "responses"

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": "Please help, bank error"},
        },
    )

    caplog.set_level("INFO", logger="backend.ai.note_style_stage")
    caplog.set_level("INFO", logger="backend.ai.note_style_results")
    build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)
    caplog.clear()
    caplog.set_level("INFO", logger="backend.ai.note_style_results")

    stage_calls: list[tuple[str, Path | str | None]] = []
    barrier_calls: list[str] = []
    reconcile_calls: list[tuple[str, Path | str | None]] = []

    def _fake_refresh(sid_arg: str, runs_root: Path | str | None = None) -> None:
        stage_calls.append((sid_arg, runs_root))

    def _fake_barriers(sid_arg: str) -> None:
        barrier_calls.append(sid_arg)

    def _fake_reconcile(sid_arg: str, runs_root: Path | str | None = None) -> dict[str, bool]:
        reconcile_calls.append((sid_arg, runs_root))
        return {}

    monkeypatch.setattr(
        "backend.ai.note_style_results.refresh_note_style_stage_from_index", _fake_refresh
    )
    monkeypatch.setattr(
        "backend.ai.note_style_results.runflow_barriers_refresh", _fake_barriers
    )
    monkeypatch.setattr(
        "backend.ai.note_style_results.reconcile_umbrella_barriers", _fake_reconcile
    )

    result_payload = {
        "sid": sid,
        "account_id": account_id,
        "analysis": {
            "tone": {"value": "conciliatory", "confidence": 0.7, "risk_flags": []},
            "context_hints": {"values": ["lender_fault"], "confidence": 0.5, "risk_flags": []},
            "emphasis": {"values": ["support_request"], "confidence": 0.6, "risk_flags": []},
        },
        "prompt_salt": "salt123",
        "note_hash": "deadbeef",
    }

    completed_at = "2024-02-02T00:00:00Z"
    result_path = store_note_style_result(
        sid,
        account_id,
        result_payload,
        runs_root=runs_root,
        completed_at=completed_at,
    )

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "backend.ai.note_style_results"
    ]

    assert any("STYLE_RESULTS_WRITTEN" in message for message in messages)
    assert any(
        "STYLE_INDEX_UPDATED" in message and "status=completed" in message
        for message in messages
    )
    assert any("STYLE_STAGE_REFRESH" in message for message in messages)
    assert any("[Runflow] Umbrella barriers:" in message for message in messages)

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)

    assert result_path == account_paths.result_file
    stored_lines = [
        line
        for line in account_paths.result_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(stored_lines) == 1
    assert json.loads(stored_lines[0]) == result_payload

    index_payload = json.loads(paths.index_file.read_text(encoding="utf-8"))
    packs = index_payload["packs"]
    assert len(packs) == 1
    entry = packs[0]
    assert entry["status"] == "completed"
    assert entry["completed_at"] == completed_at
    assert entry["result"] == account_paths.result_file.relative_to(paths.base).as_posix()
    assert entry.get("pack") == account_paths.pack_file.relative_to(paths.base).as_posix()
    totals = index_payload.get("totals")
    assert totals == {"total": 1, "completed": 1, "failed": 0}

    assert stage_calls == [(sid, runs_root)]
    assert barrier_calls == [sid]
    assert reconcile_calls == [(sid, runs_root)]
