import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from backend import config
from backend.ai.note_style import prepare_and_send
from backend.ai.note_style_sender import send_note_style_packs_for_sid
from backend.ai.note_style.schema import NoteStyleResult
from backend.config.note_style import NoteStyleResponseMode
from backend.core.ai.paths import ensure_note_style_account_paths, ensure_note_style_paths
from backend.pipeline import runs as pipeline_runs


class _StubClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def chat_completion(self, *, model, messages, temperature, **kwargs):  # type: ignore[override]
        call_index = len(self.calls)
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "kwargs": kwargs,
            }
        )

        if call_index == 0:
            payload = {
                "analysis": {
                    "tone": "Warm",
                    "context_hints": {
                        "timeframe": {"month": "April", "relative": "Last month"},
                        "topic": "Payment_Dispute",
                        "entities": {"creditor": "Example Bank", "amount": 123.45},
                    },
                    "emphasis": ["already_paid", "support_request"],
                    "confidence": 0.91,
                    "risk_flags": ["follow_up"],
                }
            }
            return {"choices": [{"message": {"content": json.dumps(payload)}}]}

        if call_index == 1:
            payload = {
                "analysis": {
                    "tone": "Calm",
                    "context_hints": {
                        "timeframe": {"month": "May", "relative": "Two months ago"},
                        "topic": "Billing_Error",
                        "entities": {"creditor": "Another Lender", "amount": 456.78},
                    },
                    "emphasis": ["billing_issue"],
                    "confidence": 0.88,
                    "risk_flags": ["escalate"],
                }
            }
            return {
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "submit_note_style_analysis",
                                        "arguments": json.dumps(payload),
                                    }
                                }
                            ],
                        }
                    }
                ]
            }

        return {"choices": [{"message": {"content": "not-json"}}]}


def _write_account_artifacts(run_dir: Path, account_id: str) -> None:
    account_dir = run_dir / "cases" / "accounts" / account_id
    account_dir.mkdir(parents=True, exist_ok=True)

    meta_payload = {"heading_guess": "Example Creditor"}
    bureaus_payload = {
        "transunion": {
            "account_type": "Credit Card",
            "account_status": "Open",
            "payment_status": "Current",
        }
    }
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


def _write_response(run_dir: Path, sid: str, account_id: str, note_text: str) -> None:
    response_dir = run_dir / "frontend" / "review" / "responses"
    response_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "sid": sid,
        "account_id": account_id,
        "answers": {"explanation": note_text},
    }
    (response_dir / f"{account_id}.result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _write_manifest(run_dir: Path, account_ids: list[str]) -> None:
    accounts_payload: dict[str, Mapping[str, str]] = {}
    for account_id in account_ids:
        accounts_payload[account_id] = {
            "dir": f"cases/accounts/{account_id}",
            "meta": "meta.json",
            "bureaus": "bureaus.json",
            "tags": "tags.json",
        }
    manifest_payload = {"artifacts": {"cases": {"accounts": accounts_payload}}}
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


@pytest.mark.parametrize("mode", ["json", "tool"])
def test_note_style_json_persistence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    sid = "SID900"
    account_ids = ["idx-900", "idx-901", "idx-902"]
    run_dir = tmp_path / sid

    for index, account_id in enumerate(account_ids, start=1):
        _write_account_artifacts(run_dir, account_id)
        _write_response(run_dir, sid, account_id, f"Customer note text #{index}.")
    _write_manifest(run_dir, account_ids)

    response_mode = NoteStyleResponseMode.JSON if mode == "json" else NoteStyleResponseMode.TOOL
    monkeypatch.setattr(config, "NOTE_STYLE_ENABLED", True)
    monkeypatch.setattr(config, "NOTE_STYLE_AUTOSEND", True)
    monkeypatch.setattr(config, "NOTE_STYLE_RESPONSE_MODE", response_mode)
    monkeypatch.setattr("backend.config.note_style.NOTE_STYLE_RESPONSE_MODE", response_mode)
    monkeypatch.setattr(config, "NOTE_STYLE_INVALID_RESULT_RETRY_ATTEMPTS", 0)
    monkeypatch.setattr(config, "NOTE_STYLE_INVALID_RESULT_RETRY_TOOL_CALL", False)
    monkeypatch.setattr(config, "NOTE_STYLE_SKIP_IF_RESULT_EXISTS", False)

    scheduled_calls: list[dict[str, Any]] = []

    def _fake_schedule_send(
        sid_arg: str,
        *,
        runs_root: Path | str | None = None,
        trigger: str | None = None,
        account_ids: tuple[str, ...] | None = None,
    ) -> None:
        scheduled_calls.append(
            {
                "sid": sid_arg,
                "runs_root": runs_root,
                "trigger": trigger,
                "account_ids": account_ids,
            }
        )

    monkeypatch.setattr("backend.ai.note_style.schedule_send_for_sid", _fake_schedule_send)

    stub_client = _StubClient()
    monkeypatch.setattr("backend.ai.note_style_sender.get_ai_client", lambda: stub_client)

    prepare_result = prepare_and_send(sid, runs_root=tmp_path)
    assert prepare_result["packs_built"] == 3
    assert scheduled_calls
    assert scheduled_calls[0]["sid"] == sid

    paths = ensure_note_style_paths(tmp_path, sid, create=False)

    runflow_path = run_dir / "runflow.json"
    runflow_payload = json.loads(runflow_path.read_text(encoding="utf-8"))
    stage_before = runflow_payload["stages"]["note_style"]
    assert stage_before["status"] == "built"
    assert stage_before["results"]["completed"] == 0
    assert stage_before.get("ready", False) is False
    assert not pipeline_runs.all_note_style_results_terminal(sid, runs_root=tmp_path)
    for account_id in account_ids:
        assert not pipeline_runs.account_result_ready(sid, account_id, runs_root=tmp_path)

    processed_accounts = send_note_style_packs_for_sid(sid, runs_root=tmp_path)
    assert set(processed_accounts) == {account_ids[0], account_ids[1]}
    assert len(stub_client.calls) == 3

    manifest_path = run_dir / "manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage_status = manifest_payload["ai"]["status"]["note_style"]
    assert stage_status["built"] is True
    assert stage_status["sent"] is True
    assert stage_status["failed"] is True
    assert isinstance(stage_status["completed_at"], str)

    runflow_payload = json.loads(runflow_path.read_text(encoding="utf-8"))
    stage_after = runflow_payload["stages"]["note_style"]
    assert stage_after["status"] in {"success", "failed"}
    assert stage_after["status"] != "built"
    results_payload = stage_after.get("results", {})
    summary_payload = stage_after.get("summary", {})
    total_results = results_payload.get("results_total") or summary_payload.get("results_total")
    assert total_results == 3
    assert results_payload.get("completed") == 2
    assert results_payload.get("failed") == 1
    assert stage_status["failed"] is True

    assert pipeline_runs.all_note_style_results_terminal(sid, runs_root=tmp_path)

    account_paths_lookup = {
        account_id: ensure_note_style_account_paths(paths, account_id, create=False)
        for account_id in account_ids
    }

    for account_id in account_ids[:2]:
        result_path = account_paths_lookup[account_id].result_file
        assert result_path.name.endswith(".jsonl")
        lines = [line for line in result_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert lines
        payload = json.loads(lines[0])
        result = NoteStyleResult.model_validate(payload)
        assert result.sid == sid
        assert result.account_id == account_id

    failure_path = account_paths_lookup[account_ids[2]].result_file
    assert failure_path.name.endswith(".jsonl")
    failure_lines = [
        line for line in failure_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert failure_lines
    failure_payload = json.loads(failure_lines[0])
    assert failure_payload["status"] == "failed"
    assert failure_payload["error"] == "invalid_result"
    assert failure_payload["sid"] == sid
    assert failure_payload["account_id"] == account_ids[2]

    for account_id in account_ids:
        assert pipeline_runs.account_result_ready(sid, account_id, runs_root=tmp_path)
