import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.ai.note_style_results import _refresh_after_index_update, store_note_style_result
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

    refresh_event = next(
        (
            entry
            for entry in structured_records
            if entry.get("event") == "NOTE_STYLE_REFRESH"
        ),
        None,
    )
    assert refresh_event is not None
    assert refresh_event.get("packs_expected") == 1

    assert result_path == account_paths.result_file
    stored_lines = [
        line
        for line in account_paths.result_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(stored_lines) == 1
    stored_payload = json.loads(stored_lines[0])
    assert {
        "sid",
        "account_id",
        "evaluated_at",
        "analysis",
        "note_metrics",
    }.issubset(stored_payload.keys())
    assert stored_payload["evaluated_at"] == completed_at
    assert stored_payload["note_metrics"] == baseline_metrics
    assert set(stored_payload["note_metrics"].keys()) == {"char_len", "word_len"}
    assert stored_payload["sid"] == sid
    assert stored_payload["account_id"] == account_id
    note_hash_value = stored_payload.get("note_hash")
    if note_hash_value is not None:
        assert isinstance(note_hash_value, str)
        assert note_hash_value.strip()
    assert "prompt_salt" not in stored_payload
    assert "fingerprint_hash" not in stored_payload
    analysis = stored_payload["analysis"]
    assert analysis["tone"] == "Empathetic"
    assert analysis["confidence"] == 0.5
    assert analysis["emphasis"] == ["support_request", "evidence_provided"]
    context = analysis["context_hints"]
    assert context["topic"] == "billing_error"
    assert context["timeframe"] == {"month": 1, "relative": "last_year"}
    assert context["entities"] == {"creditor": "Capital Bank", "amount": 123.45}
    assert analysis["risk_flags"] == ["needs_review", "follow_up"]

    index_payload = json.loads(paths.index_file.read_text(encoding="utf-8"))
    packs = index_payload["packs"]
    assert len(packs) == 1
    entry = packs[0]
    assert entry["status"] == "completed"
    note_hash_entry = entry.get("note_hash")
    if note_hash_entry is not None:
        assert isinstance(note_hash_entry, str)
        assert note_hash_entry.strip()

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
        "note_hash",
    }
    assert stored_payload["analysis"]["tone"] == "neutral"
    assert isinstance(stored_payload["note_hash"], str)
    assert stored_payload["note_hash"].strip()
    assert "prompt_salt" not in stored_payload
    assert stored_payload["evaluated_at"].endswith("Z")
    assert set(stored_payload["note_metrics"].keys()) == {"char_len", "word_len"}


def test_store_note_style_result_writes_failure_on_validation_error(
    tmp_path: Path,
) -> None:
    sid = "SID910"
    account_id = "idx-910"
    runs_root = tmp_path / "runs"
    response_dir = runs_root / sid / "frontend" / "review" / "responses"

    run_dir = runs_root / sid
    _write_manifest(run_dir, account_id)

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"note": "sample"},
        },
    )

    build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)

    invalid_payload = {
        "sid": sid,
        "account_id": account_id,
        "analysis": None,
        "note_metrics": {"char_len": 12, "word_len": 3},
    }

    with pytest.raises(ValueError) as excinfo:
        store_note_style_result(
            sid,
            account_id,
            invalid_payload,
            runs_root=runs_root,
            update_index=False,
        )

    assert "validation_error" in str(excinfo.value)

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)
    stored_lines = [
        line
        for line in account_paths.result_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(stored_lines) == 1
    failure_payload = json.loads(stored_lines[0])
    assert failure_payload["status"] == "failed"
    assert failure_payload["error"] == "validation_error"
    assert failure_payload["sid"] == sid
    assert failure_payload["account_id"] == account_id
    assert failure_payload["details"]
    assert failure_payload["evaluated_at"].endswith("Z")


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
    assert stage_status["sent"] is False
    assert stage_status["completed_at"] is None

def test_refresh_after_index_update_defers_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    sid = "SID-success-guard"
    runs_root = tmp_path / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    paths = ensure_note_style_paths(runs_root, sid, create=True)

    stage_updates: list[dict[str, object]] = []
    record_calls: list[dict[str, object]] = []

    def _fake_update(
        sid_arg: str,
        *,
        state: str | None = None,
        sent: bool | None = None,
        completed_at: str | None | object = None,
        **_: object,
    ) -> None:
        stage_updates.append(
            {
                "sid": sid_arg,
                "state": state,
                "sent": sent,
                "completed_at": completed_at,
            }
        )

    def _fake_record(
        sid_arg: str,
        stage: str,
        *,
        status: str | None = None,
        **_: object,
    ) -> None:
        record_calls.append({"sid": sid_arg, "stage": stage, "status": status})

    stage_view = SimpleNamespace(
        total_expected=3,
        completed_total=1,
        failed_total=0,
        built_complete=True,
        state="success",
        is_terminal=True,
    )

    monkeypatch.setattr(
        "backend.ai.note_style_results.refresh_note_style_stage_from_results",
        lambda *args, **kwargs: (3, 1, 0),
    )
    monkeypatch.setattr(
        "backend.ai.note_style_results.reconcile_umbrella_barriers",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "backend.ai.note_style_results.note_style_stage_view",
        lambda *args, **kwargs: stage_view,
    )
    monkeypatch.setattr(
        "backend.ai.note_style_results.update_note_style_stage_status",
        _fake_update,
    )
    monkeypatch.setattr(
        "backend.ai.note_style_results.record_stage",
        _fake_record,
    )
    monkeypatch.setattr(
        "backend.ai.note_style_results.runflow_barriers_refresh",
        lambda *args, **kwargs: None,
    )

    caplog.set_level("INFO", logger="backend.ai.note_style_results")

    _refresh_after_index_update(
        sid=sid,
        account_id="acct-guard",
        runs_root_path=runs_root,
        paths=paths,
        updated_entry={"status": "completed"},
        totals={"total": 3, "completed": 1, "failed": 0},
        skipped_count=0,
    )

    assert stage_updates, "expected stage update to be recorded"
    update_payload = stage_updates[-1]
    assert update_payload["state"] == "in_progress"
    assert update_payload["sent"] is False
    assert update_payload["completed_at"] is None

    assert record_calls, "expected record_stage call"
    record_payload = record_calls[-1]
    assert record_payload["status"] == "in_progress"

    messages = [record.getMessage() for record in caplog.records]
    assert any("NOTE_STYLE_SUCCESS_DEFERRED" in message for message in messages)
