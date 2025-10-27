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


def _seed_account_context(run_dir: Path, account_id: str) -> None:
    account_dir = run_dir / "cases" / "accounts" / account_id
    account_dir.mkdir(parents=True, exist_ok=True)
    meta_payload = {"heading_guess": f"Account {account_id}"}
    bureaus_payload = {"transunion": {"account_type": "Credit Card"}}
    tags_payload = [{"kind": "issue", "type": "late_payment"}]

    (account_dir / "meta.json").write_text(
        json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (account_dir / "bureaus.json").write_text(
        json.dumps(bureaus_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (account_dir / "tags.json").write_text(
        json.dumps(tags_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


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
    note_text = pack_payload["note_text"]
    baseline_metrics = {
        "char_len": len(note_text),
        "word_len": len(note_text.split()),
    }

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
    assert set(stored_payload.keys()) == {
        "sid",
        "account_id",
        "evaluated_at",
        "analysis",
        "note_metrics",
    }
    assert stored_payload["evaluated_at"] == completed_at
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
    assert set(stored_payload.keys()) == {
        "sid",
        "account_id",
        "evaluated_at",
        "analysis",
        "note_metrics",
    }
    assert stored_payload["analysis"]["tone"] == "neutral"
    assert "note_hash" not in stored_payload
    assert "prompt_salt" not in stored_payload
    assert stored_payload["evaluated_at"].endswith("Z")
    assert set(stored_payload["note_metrics"].keys()) == {"char_len", "word_len"}


def test_manifest_status_tracks_partial_and_complete_results(tmp_path: Path) -> None:
    sid = "SID905"
    runs_root = tmp_path / "runs"
    run_dir = runs_root / sid
    account_ids = ["idx-905a", "idx-905b"]

    manifest_payload = {
        "artifacts": {
            "cases": {
                "accounts": {
                    account_id: {
                        "dir": f"cases/accounts/{account_id}",
                        "meta": "meta.json",
                        "bureaus": "bureaus.json",
                        "tags": "tags.json",
                    }
                    for account_id in account_ids
                }
            }
        }
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    responses_dir = runs_root / sid / "frontend" / "review" / "responses"
    for account_id in account_ids:
        _seed_account_context(run_dir, account_id)
        _write_response(
            responses_dir / f"{account_id}.result.json",
            {
                "sid": sid,
                "account_id": account_id,
                "answers": {"explanation": f"Support needed for {account_id}"},
            },
        )

    for account_id in account_ids:
        build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    index_payload = json.loads(paths.index_file.read_text(encoding="utf-8"))
    pack_accounts = {entry["account_id"] for entry in index_payload["packs"]}
    assert pack_accounts == set(account_ids)

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage_status = manifest_data["ai"]["status"]["note_style"]
    assert stage_status["built"] is True
    assert stage_status["sent"] is False
    assert stage_status["completed_at"] is None

    first_payload = {
        "sid": sid,
        "account_id": account_ids[0],
        "analysis": {
            "tone": "supportive",
            "context_hints": {
                "topic": "billing_error",
                "timeframe": {"month": "2024-03-01", "relative": "recent"},
                "entities": {"creditor": "Capital", "amount": 125.0},
            },
            "emphasis": ["support_request"],
            "confidence": 0.72,
            "risk_flags": [],
        },
    }
    store_note_style_result(
        sid,
        account_ids[0],
        first_payload,
        runs_root=runs_root,
    )

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage_status = manifest_data["ai"]["status"]["note_style"]
    assert stage_status["built"] is True
    assert stage_status["sent"] is True
    assert stage_status["completed_at"] is not None
    assert stage_status["completed_at"].endswith("Z")

    second_payload = {
        "sid": sid,
        "account_id": account_ids[1],
        "analysis": {
            "tone": "empathetic",
            "context_hints": {
                "topic": "payment_issue",
                "timeframe": {"month": "2024-04-01", "relative": "later"},
                "entities": {"creditor": "Capital", "amount": 87.0},
            },
            "emphasis": ["follow_up"],
            "confidence": 0.68,
            "risk_flags": ["needs_review"],
        },
    }
    store_note_style_result(
        sid,
        account_ids[1],
        second_payload,
        runs_root=runs_root,
        completed_at="2025-01-02T00:00:00Z",
    )

    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage_status = manifest_data["ai"]["status"]["note_style"]
    assert stage_status["built"] is True
    assert stage_status["sent"] is True
    assert stage_status["completed_at"] is not None
    assert stage_status["completed_at"].endswith("Z")
