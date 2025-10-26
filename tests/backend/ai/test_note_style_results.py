import json
import logging
from pathlib import Path

import pytest

from backend.ai.note_style_results import store_note_style_result
from backend.ai.note_style_stage import build_note_style_pack_for_account
from backend.core.ai.paths import ensure_note_style_account_paths, ensure_note_style_paths


def _write_response(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_manifest(run_dir: Path, account_id: str) -> Path:
    account_dir = run_dir / "cases" / "accounts" / account_id
    account_dir.mkdir(parents=True, exist_ok=True)
    manifest_payload = {
        "artifacts": {
            "cases": {
                "accounts": {
                    account_id: {
                        "dir": "cases/accounts/" + account_id,
                        "meta": "meta.json",
                        "bureaus": "bureaus.json",
                        "tags": "tags.json",
                    }
                }
            }
        }
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return account_dir


def test_store_note_style_result_updates_index_and_triggers_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    sid = "SID900"
    account_id = "idx-900"
    runs_root = tmp_path / "runs"
    response_dir = runs_root / sid / "frontend" / "review" / "responses"

    run_dir = runs_root / sid
    _write_manifest(run_dir, account_id)

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
        "backend.ai.note_style_results.refresh_note_style_stage_from_results", _fake_refresh
    )
    monkeypatch.setattr(
        "backend.ai.note_style_results.runflow_barriers_refresh", _fake_barriers
    )
    monkeypatch.setattr(
        "backend.ai.note_style_results.reconcile_umbrella_barriers", _fake_reconcile
    )

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)
    pack_payload = json.loads(
        account_paths.pack_file.read_text(encoding="utf-8").splitlines()[0]
    )
    context_payload = pack_payload["context"]
    assert "note_metrics" not in pack_payload
    note_text = context_payload["note_text"]
    baseline_metrics = {"char_len": len(note_text), "word_len": len(note_text.split())}

    result_payload = {
        "sid": sid,
        "account_id": account_id,
        "analysis": {
            "tone": "Empathetic",
            "context_hints": {
                "timeframe": {"month": "2024-01-01", "relative": "Last Year"},
                "topic": "Billing Error",
                "entities": {"creditor": "Capital Bank", "amount": "123.45"},
            },
            "emphasis": ["Support Request", "Evidence Provided"],
            "confidence": 0.82,
            "risk_flags": ["Needs Review", "Follow Up"],
        },
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

    assert any("NOTE_STYLE_PARSED" in message for message in messages)
    assert any(
        "NOTE_STYLE_INDEX_UPDATED" in message and "status=completed" in message
        for message in messages
    )
    assert any("NOTE_STYLE_REFRESH" in message for message in messages)
    assert any(
        "NOTE_STYLE_STAGE_REFRESH_DETAIL" in message for message in messages
    )
    assert any("[Runflow] Umbrella barriers:" in message for message in messages)

    structured_records = [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == "backend.ai.note_style_results"
        and record.getMessage().startswith("{")
    ]
    structured_event = next(
        (
            entry
            for entry in structured_records
            if entry.get("event") == "NOTE_STYLE_PARSED"
        ),
        None,
    )
    assert structured_event is not None
    assert structured_event.get("note_metrics")
    assert structured_event.get("risk_flags") is not None

    assert result_path == account_paths.result_file
    stored_lines = [
        line
        for line in account_paths.result_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(stored_lines) == 1
    stored_payload = json.loads(stored_lines[0])
    assert set(stored_payload.keys()) == {"sid", "account_id", "analysis", "note_metrics"}
    assert "evaluated_at" not in stored_payload
    assert stored_payload["note_metrics"] == baseline_metrics
    assert set(stored_payload["note_metrics"].keys()) == {"char_len", "word_len"}
    assert stored_payload["sid"] == sid
    assert stored_payload["account_id"] == account_id
    assert "note_hash" not in stored_payload
    assert "prompt_salt" not in stored_payload
    assert "fingerprint_hash" not in stored_payload
    analysis = stored_payload["analysis"]
    assert analysis["tone"] == "Empathetic"
    assert analysis["confidence"] == 0.5
    assert analysis["emphasis"] == ["support_request", "evidence_provided"]
    context = analysis["context_hints"]
    assert context["topic"] == "billing_error"
    assert context["timeframe"] == {"month": "2024-01-01", "relative": "last_year"}
    assert context["entities"] == {"creditor": "Capital Bank", "amount": 123.45}
    assert analysis["risk_flags"] == ["needs_review", "follow_up"]

    index_payload = json.loads(paths.index_file.read_text(encoding="utf-8"))
    packs = index_payload["packs"]
    assert len(packs) == 1
    entry = packs[0]
    assert entry["status"] == "completed"
    assert "note_hash" not in entry

    assert stage_calls == [(sid, runs_root)]
    assert barrier_calls == [sid]
    assert reconcile_calls == [(sid, runs_root)]


def test_store_note_style_result_handles_short_note(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    sid = "SID901"
    account_id = "idx-901"
    runs_root = tmp_path / "runs"
    response_dir = runs_root / sid / "frontend" / "review" / "responses"

    run_dir = runs_root / sid
    _write_manifest(run_dir, account_id)

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"note": "hi"},
        },
    )

    build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)

    result_payload = {
        "sid": sid,
        "account_id": account_id,
        "analysis": {
            "tone": "neutral",
            "context_hints": {
                "topic": "other",
                "timeframe": {"month": None, "relative": None},
                "entities": {"creditor": None, "amount": None},
            },
            "emphasis": [],
            "confidence": 0.2,
            "risk_flags": [],
        },
    }

    with caplog.at_level(logging.INFO, logger="backend.ai.note_style_results"):
        store_note_style_result(
            sid,
            account_id,
            result_payload,
            runs_root=runs_root,
        )

    stored_payload = json.loads(account_paths.result_file.read_text(encoding="utf-8"))
    assert set(stored_payload.keys()) == {"sid", "account_id", "analysis", "note_metrics"}
    assert stored_payload["analysis"]["tone"] == "neutral"
    assert "note_hash" not in stored_payload
    assert "prompt_salt" not in stored_payload
    assert "evaluated_at" not in stored_payload
    assert set(stored_payload["note_metrics"].keys()) == {"char_len", "word_len"}
