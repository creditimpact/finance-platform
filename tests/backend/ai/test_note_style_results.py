from __future__ import annotations

import hashlib
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

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)
    baseline_payload = json.loads(account_paths.result_file.read_text(encoding="utf-8"))
    baseline_metrics = baseline_payload["note_metrics"]

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

    assert any("NOTE_STYLE_RESULT_WRITTEN" in message for message in messages)
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
            if entry.get("event") == "NOTE_STYLE_RESULTS_WRITTEN"
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
    assert stored_payload["note_metrics"] == baseline_metrics
    assert stored_payload["sid"] == sid
    assert stored_payload["account_id"] == account_id
    assert stored_payload["note_hash"] == result_payload["note_hash"]
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
    assert entry["completed_at"] == completed_at
    assert (
        entry["result_path"]
        == account_paths.result_file.relative_to(paths.base).as_posix()
    )
    assert entry.get("pack") == account_paths.pack_file.relative_to(paths.base).as_posix()
    expected_note_hash = hashlib.sha256("Please help, bank error".encode("utf-8")).hexdigest()
    assert entry["note_hash"] == expected_note_hash
    totals = index_payload.get("totals")
    assert totals == {"total": 1, "completed": 1, "failed": 0}

    assert stage_calls == [(sid, runs_root)]
    assert barrier_calls == [sid]
    assert reconcile_calls == [(sid, runs_root)]


def test_store_note_style_result_normalizes_analysis_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "SID905"
    account_id = "idx-905"
    runs_root = tmp_path / "runs"
    response_dir = runs_root / sid / "frontend" / "review" / "responses"

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": "Please review my dispute."},
        },
    )

    build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)

    monkeypatch.setattr(
        "backend.ai.note_style_results.runflow_barriers_refresh", lambda _sid: None
    )
    monkeypatch.setattr(
        "backend.ai.note_style_results.reconcile_umbrella_barriers",
        lambda _sid, runs_root=None: {},
    )

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)
    baseline_payload = json.loads(account_paths.result_file.read_text(encoding="utf-8"))
    prompt_salt = baseline_payload["prompt_salt"]
    note_hash = baseline_payload["note_hash"]
    evaluated_at = baseline_payload["evaluated_at"]
    fingerprint_hash = baseline_payload["fingerprint_hash"]

    result_payload = {
        "sid": sid,
        "account_id": account_id,
        "analysis": {
            "tone": {"value": "Assertive", "confidence": 1.2, "risk_flags": ["Legal Threat"]},
            "context_hints": {
                "topic": "Billing Error",
                "timeframe": {"month": "March 5, 2024", "relative": "Last Year"},
                "entities": {"creditor": "Capital Bank", "amount": "120"},
                "risk_flags": ["Needs Review"],
            },
            "emphasis": {
                "values": [
                    "Support Request",
                    "Fee Waiver",
                    "Identity Concerns",
                    "Paid Already",
                    "Ownership Dispute",
                    "Evidence Provided",
                    "Extra Item",
                ],
                "risk_flags": ["Personal Data"],
            },
            "confidence": 1.25,
            "risk_flags": ["Escalation Risk", "Escalation Risk", "Needs Review"],
        },
        "prompt_salt": baseline_payload["prompt_salt"],
        "note_hash": baseline_payload["note_hash"],
        "note_metrics": baseline_payload["note_metrics"],
        "evaluated_at": baseline_payload["evaluated_at"],
        "fingerprint_hash": baseline_payload["fingerprint_hash"],
    }

    store_note_style_result(
        sid,
        account_id,
        result_payload,
        runs_root=runs_root,
    )

    stored_payload = json.loads(account_paths.result_file.read_text(encoding="utf-8"))
    analysis = stored_payload["analysis"]

    assert analysis["tone"] == "Assertive"
    assert analysis["confidence"] == 0.5
    assert analysis["risk_flags"] == ["escalation_risk", "needs_review"]

    context = analysis["context_hints"]
    assert context["topic"] == "billing_error"
    timeframe = context["timeframe"]
    assert timeframe["month"] == "2024-03-05"
    assert timeframe["relative"] == "last_year"
    assert context["entities"] == {"creditor": "Capital Bank", "amount": 120.0}

    emphasis = analysis["emphasis"]
    assert emphasis == [
        "support_request",
        "fee_waiver",
        "identity_concerns",
        "paid_already",
        "ownership_dispute",
        "evidence_provided",
    ]


def test_store_note_style_result_requires_all_accounts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "SID901"
    account_primary = "idx-901A"
    account_secondary = "idx-901B"
    runs_root = tmp_path / "runs"
    response_dir = runs_root / sid / "frontend" / "review" / "responses"

    _write_response(
        response_dir / f"{account_primary}.result.json",
        {
            "sid": sid,
            "account_id": account_primary,
            "answers": {"explanation": "We resolved this with the bank."},
        },
    )
    _write_response(
        response_dir / f"{account_secondary}.result.json",
        {
            "sid": sid,
            "account_id": account_secondary,
            "answers": {"explanation": "Customer note requires follow-up."},
        },
    )

    build_note_style_pack_for_account(sid, account_primary, runs_root=runs_root)
    build_note_style_pack_for_account(sid, account_secondary, runs_root=runs_root)

    monkeypatch.setattr(
        "backend.ai.note_style_results.runflow_barriers_refresh", lambda _sid: None
    )
    monkeypatch.setattr(
        "backend.ai.note_style_results.reconcile_umbrella_barriers",
        lambda _sid, runs_root=None: {},
    )

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    primary_paths = ensure_note_style_account_paths(
        paths, account_primary, create=False
    )
    secondary_paths = ensure_note_style_account_paths(
        paths, account_secondary, create=False
    )

    primary_payload = json.loads(primary_paths.result_file.read_text(encoding="utf-8"))
    primary_payload["analysis"] = {
        "tone": "positive",
        "context_hints": {
            "timeframe": {"month": None, "relative": None},
            "topic": "other",
            "entities": {"creditor": None, "amount": None},
        },
        "emphasis": [],
        "confidence": 0.9,
        "risk_flags": [],
    }
    store_note_style_result(
        sid,
        account_primary,
        primary_payload,
        runs_root=runs_root,
    )

    runflow_path = runs_root / sid / "runflow.json"
    runflow_payload = json.loads(runflow_path.read_text(encoding="utf-8"))
    stage_payload = runflow_payload["stages"]["note_style"]
    assert stage_payload["status"] == "built"
    assert stage_payload["results"]["results_total"] == 2
    assert stage_payload["results"]["completed"] == 1
    assert stage_payload["results"]["failed"] == 0

    index_payload = json.loads(paths.index_file.read_text(encoding="utf-8"))
    statuses = {entry["account_id"]: entry["status"] for entry in index_payload["packs"]}
    assert statuses[account_primary] == "completed"
    assert statuses[account_secondary] == "built"

    secondary_payload = json.loads(
        secondary_paths.result_file.read_text(encoding="utf-8")
    )
    secondary_payload["analysis"] = {
        "tone": "neutral",
        "context_hints": {
            "timeframe": {"month": None, "relative": None},
            "topic": "other",
            "entities": {"creditor": None, "amount": None},
        },
        "emphasis": [],
        "confidence": 0.8,
        "risk_flags": [],
    }
    store_note_style_result(
        sid,
        account_secondary,
        secondary_payload,
        runs_root=runs_root,
    )

    updated_payload = json.loads(runflow_path.read_text(encoding="utf-8"))
    updated_stage = updated_payload["stages"]["note_style"]
    assert updated_stage["status"] == "success"
    assert updated_stage["results"]["results_total"] == 2
    assert updated_stage["results"]["completed"] == 2
    assert updated_stage["results"]["failed"] == 0


def test_store_note_style_result_logs_missing_fields(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    sid = "SID990"
    account_id = "idx-990"
    runs_root = tmp_path / "runs"
    response_dir = runs_root / sid / "frontend" / "review" / "responses"

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": "Short note"},
        },
    )

    build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)

    caplog.set_level("WARNING", logger="backend.ai.note_style_results")

    result_payload = {
        "sid": sid,
        "account_id": account_id,
        "analysis": {
            "tone": "",
            "context_hints": {},
            "confidence": None,
        },
        "prompt_salt": "salt",
        "note_hash": "hash",
    }

    with pytest.raises(ValueError):
        store_note_style_result(sid, account_id, result_payload, runs_root=runs_root)

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    log_path = paths.log_file
    assert log_path.exists()
    contents = log_path.read_text(encoding="utf-8")
    assert "guard_failed" in contents
    assert account_id in contents

    warnings = [
        record.getMessage()
        for record in caplog.records
        if record.name == "backend.ai.note_style_results"
    ]
    assert any("NOTE_STYLE_RESULT_GUARD_FAILED" in message for message in warnings)


def test_store_note_style_result_warns_on_fingerprint_mismatch(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    sid = "SID991"
    account_id = "idx-991"
    runs_root = tmp_path / "runs"
    response_dir = runs_root / sid / "frontend" / "review" / "responses"

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": "Account paid in full."},
        },
    )

    build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)

    pack_payload = json.loads(
        account_paths.pack_file.read_text(encoding="utf-8").splitlines()[0]
    )
    pack_payload["fingerprint_hash"] = "pack-hash"
    account_paths.pack_file.write_text(
        json.dumps(pack_payload, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    baseline_result = json.loads(
        account_paths.result_file.read_text(encoding="utf-8").splitlines()[0]
    )

    result_payload = {
        "sid": sid,
        "account_id": account_id,
        "analysis": {
            "tone": "neutral",
            "context_hints": {
                "topic": "other",
                "timeframe": {"month": None, "relative": None},
                "entities": {"creditor": None, "amount": None},
                "risk_flags": [],
            },
            "emphasis": [],
            "confidence": 0.25,
            "risk_flags": [],
        },
        "prompt_salt": baseline_result["prompt_salt"],
        "note_hash": baseline_result["note_hash"],
        "note_metrics": baseline_result["note_metrics"],
        "evaluated_at": baseline_result["evaluated_at"],
        "fingerprint_hash": "result-hash",
    }

    caplog.set_level("WARNING", logger="backend.ai.note_style_results")
    store_note_style_result(
        sid,
        account_id,
        result_payload,
        runs_root=runs_root,
    )

    structured_warnings = [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == "backend.ai.note_style_results"
        and record.levelno >= logging.WARNING
        and record.getMessage().startswith("{")
    ]

    assert any(
        entry.get("event") == "NOTE_STYLE_FINGERPRINT_MISMATCH"
        and entry.get("pack_fingerprint_hash") == "pack-hash"
        and entry.get("result_fingerprint_hash") == "result-hash"
        for entry in structured_warnings
    )


def test_store_note_style_result_clamps_confidence_for_short_notes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "SID910"
    account_id = "idx-910"
    runs_root = tmp_path / "runs"
    response_dir = runs_root / sid / "frontend" / "review" / "responses"

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": "Need refund now"},
        },
    )

    build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)

    monkeypatch.setattr(
        "backend.ai.note_style_results.runflow_barriers_refresh", lambda _sid: None
    )
    monkeypatch.setattr(
        "backend.ai.note_style_results.reconcile_umbrella_barriers",
        lambda _sid, runs_root=None: {},
    )

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)
    baseline_payload = json.loads(account_paths.result_file.read_text(encoding="utf-8"))
    prompt_salt = baseline_payload["prompt_salt"]
    note_hash = baseline_payload["note_hash"]
    evaluated_at = baseline_payload["evaluated_at"]
    fingerprint_hash = baseline_payload["fingerprint_hash"]

    result_payload = {
        "sid": sid,
        "account_id": account_id,
        "analysis": {
            "tone": "assertive",
            "context_hints": {
                "timeframe": {"month": None, "relative": None},
                "topic": "payment_dispute",
                "entities": {"creditor": None, "amount": None},
            },
            "emphasis": ["support_request"],
            "confidence": 0.92,
            "risk_flags": [],
        },
        "prompt_salt": prompt_salt,
        "note_hash": note_hash,
        "evaluated_at": evaluated_at,
        "fingerprint_hash": fingerprint_hash,
    }

    store_note_style_result(
        sid,
        account_id,
        result_payload,
        runs_root=runs_root,
        completed_at="2024-02-03T00:00:00Z",
    )

    stored_payload = json.loads(account_paths.result_file.read_text(encoding="utf-8"))
    assert stored_payload["analysis"]["confidence"] == 0.5
    assert stored_payload["note_metrics"] == baseline_payload["note_metrics"]


def test_store_note_style_result_adds_unsupported_claim_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "SID911"
    account_id = "idx-911"
    runs_root = tmp_path / "runs"
    response_dir = runs_root / sid / "frontend" / "review" / "responses"

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {
                "explanation": "They owe me $500 but I have no documents"
            },
        },
    )

    build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)

    monkeypatch.setattr(
        "backend.ai.note_style_results.runflow_barriers_refresh", lambda _sid: None
    )
    monkeypatch.setattr(
        "backend.ai.note_style_results.reconcile_umbrella_barriers",
        lambda _sid, runs_root=None: {},
    )

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)

    baseline_payload = json.loads(account_paths.result_file.read_text(encoding="utf-8"))
    prompt_salt = baseline_payload["prompt_salt"]
    note_hash = baseline_payload["note_hash"]
    evaluated_at = baseline_payload["evaluated_at"]
    fingerprint_hash = baseline_payload["fingerprint_hash"]

    result_payload = {
        "sid": sid,
        "account_id": account_id,
        "analysis": {
            "tone": "assertive",
            "context_hints": {
                "timeframe": {"month": None, "relative": None},
                "topic": "payment_dispute",
                "entities": {"creditor": None, "amount": 500},
            },
            "emphasis": [],
            "confidence": 0.66,
            "risk_flags": [],
        },
        "prompt_salt": prompt_salt,
        "note_hash": note_hash,
        "evaluated_at": evaluated_at,
        "fingerprint_hash": fingerprint_hash,
    }

    store_note_style_result(
        sid,
        account_id,
        result_payload,
        runs_root=runs_root,
        completed_at="2024-02-04T00:00:00Z",
    )

    stored_payload = json.loads(account_paths.result_file.read_text(encoding="utf-8"))
    risk_flags = stored_payload["analysis"].get("risk_flags")
    assert "unsupported_claim" in risk_flags
    assert stored_payload["analysis"]["confidence"] <= 0.5
