import json
from pathlib import Path

import pytest

from backend.ai.note_style.schema import validate_note_style_analysis

from backend.ai.note_style_stage import build_note_style_pack_for_account
from backend.ai.note_style_sender import send_note_style_packs_for_sid
from backend.core.ai.paths import ensure_note_style_account_paths, ensure_note_style_paths


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _bootstrap_note_style_run(tmp_path: Path, sid: str, account_id: str) -> Path:
    run_dir = tmp_path / sid
    account_dir = run_dir / "cases" / "accounts" / account_id
    account_dir.mkdir(parents=True, exist_ok=True)

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
                }
            }
        }
    }
    _write_json(run_dir / "manifest.json", manifest_payload)

    _write_json(
        run_dir / "frontend" / "review" / "responses" / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": "Customer already paid this bill."},
        },
    )

    _write_json(account_dir / "meta.json", {"heading_guess": "Customer Account"})
    _write_json(
        account_dir / "tags.json",
        {"tags": [{"kind": "issue", "type": "Billing"}]},
    )
    _write_json(
        account_dir / "bureaus.json",
        {
            "experian": {
                "dispute_reason": "Incorrect fee",
                "account_number": "1234",
                "creditor": "Acme Bank",
            }
        },
    )

    return run_dir


def test_note_style_pack_serializes_user_content(tmp_path: Path) -> None:
    sid = "SID200"
    account_id = "idx-200"
    _bootstrap_note_style_run(tmp_path, sid, account_id)

    build_note_style_pack_for_account(sid, account_id, runs_root=tmp_path)

    paths = ensure_note_style_paths(tmp_path, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)
    pack_lines = [
        line
        for line in account_paths.pack_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(pack_lines) == 1

    record = json.loads(pack_lines[0])
    messages = record.get("messages")
    assert isinstance(messages, list)
    assert len(messages) >= 2
    content_value = messages[1]["content"]
    assert isinstance(content_value, (str, list))
    if isinstance(content_value, str):
        json.loads(content_value)
    else:
        for item in content_value:
            assert isinstance(item, (str, dict))


class _StubHTTPError(Exception):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class _ErroringClient:
    def chat_completion(self, **_: object) -> None:
        raise _StubHTTPError("Invalid payload", 400)


class _SuccessfulClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def chat_completion(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        analysis_payload = {
            "tone": "Empathetic",
            "context_hints": {
                "timeframe": {"month": 6, "relative": "Last month"},
                "topic": "Billing",
                "entities": {
                    "creditor": "Capital One",
                    "amount": "$123.45 USD",
                },
            },
            "emphasis": ["paid_already", "support_request"],
            "confidence": "0.91",
            "risk_flags": "FOLLOW_UP",
        }
        response_payload = dict(analysis_payload)
        response_payload["analysis"] = analysis_payload
        response_payload["note"] = "Customer already paid"
        return {
            "choices": [
                {"message": {"content": json.dumps(response_payload)}}
            ]
        }


def test_note_style_sender_writes_failure_on_http_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "SID201"
    account_id = "idx-201"
    run_dir = _bootstrap_note_style_run(tmp_path, sid, account_id)

    build_note_style_pack_for_account(sid, account_id, runs_root=tmp_path)

    monkeypatch.setattr(
        "backend.ai.note_style_sender.get_ai_client", lambda: _ErroringClient()
    )

    processed = send_note_style_packs_for_sid(sid, runs_root=tmp_path)
    assert processed == []

    paths = ensure_note_style_paths(tmp_path, sid, create=False)
    failure_account_paths = ensure_note_style_account_paths(paths, account_id, create=False)
    failure_path = failure_account_paths.result_file
    assert failure_path.exists(), "expected failure artifact to be written"
    failure_payload = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure_payload["status"] == "failed"
    error_value = failure_payload.get("error")
    if isinstance(error_value, dict):
        assert error_value.get("code") == 400
    else:
        assert error_value == "Invalid payload"

    runflow_path = run_dir / "runflow.json"
    runflow_payload = json.loads(runflow_path.read_text(encoding="utf-8"))
    note_style_stage = runflow_payload["stages"]["note_style"]
    assert note_style_stage["status"] == "error"


def test_note_style_sender_records_successful_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "SID202"
    account_id = "idx-202"
    run_dir = _bootstrap_note_style_run(tmp_path, sid, account_id)

    build_note_style_pack_for_account(sid, account_id, runs_root=tmp_path)

    client = _SuccessfulClient()
    monkeypatch.setattr("backend.ai.note_style_sender.get_ai_client", lambda: client)

    processed = send_note_style_packs_for_sid(sid, runs_root=tmp_path)
    assert processed == [account_id]
    assert client.calls, "expected OpenAI client to be invoked"

    paths = ensure_note_style_paths(tmp_path, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)
    result_text = account_paths.result_file.read_text(encoding="utf-8").splitlines()[0]
    result_payload = json.loads(result_text)
    assert result_payload["analysis"]["tone"] == "Empathetic"

    runflow_path = run_dir / "runflow.json"
    runflow_payload = json.loads(runflow_path.read_text(encoding="utf-8"))
    note_style_stage = runflow_payload["stages"]["note_style"]
    assert note_style_stage["status"] == "success"


def test_note_style_sender_smoke_schema_conformance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "SID203"
    account_id = "idx-203"
    run_dir = _bootstrap_note_style_run(tmp_path, sid, account_id)

    build_note_style_pack_for_account(sid, account_id, runs_root=tmp_path)

    client = _SuccessfulClient()
    monkeypatch.setattr("backend.ai.note_style_sender.get_ai_client", lambda: client)

    processed = send_note_style_packs_for_sid(sid, runs_root=tmp_path)
    assert processed == [account_id]

    assert client.calls, "expected OpenAI client to be invoked"
    last_message = client.calls[0]["messages"][-1]
    content_value = last_message.get("content") if isinstance(last_message, dict) else None
    assert isinstance(content_value, str) and content_value.strip()

    paths = ensure_note_style_paths(tmp_path, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)
    result_lines = [
        line
        for line in account_paths.result_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert result_lines, "expected result file to contain payload"

    result_payload = json.loads(result_lines[-1])
    analysis_payload = result_payload["analysis"]
    assert "analysis" not in analysis_payload
    assert "note" not in analysis_payload

    timeframe_month = (
        analysis_payload["context_hints"]["timeframe"]["month"]
    )
    assert isinstance(timeframe_month, str)
    assert timeframe_month == "06"

    valid, errors = validate_note_style_analysis(analysis_payload)
    assert valid, errors

    runflow_payload = json.loads((run_dir / "runflow.json").read_text(encoding="utf-8"))
    note_style_stage = runflow_payload["stages"]["note_style"]
    assert note_style_stage["status"] == "success"
