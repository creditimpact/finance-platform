from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import pytest

import backend.config.note_style as note_style_config

from backend.ai.note_style_results import (
    complete_note_style_result,
    store_note_style_result,
)
from backend.ai.note_style_sender import (
    send_note_style_pack_for_account,
    send_note_style_packs_for_sid,
)
from backend.ai.note_style_stage import build_note_style_pack_for_account
from backend.core.ai.paths import ensure_note_style_account_paths, ensure_note_style_paths
from backend import config
from backend.config.note_style import NoteStyleResponseMode


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


class _StubClient:
    def __init__(self, *, response: Mapping[str, Any] | None = None) -> None:
        self.calls: list[dict[str, object]] = []
        self._analysis_payload = response or {
            "tone": "Empathetic",
            "context_hints": {
                "timeframe": {"month": "April", "relative": "Last month"},
                "topic": "Payment_Dispute",
                "entities": {"creditor": "capital one", "amount": 123.45},
            },
            "emphasis": ["paid_already", "Custom", "support_request"],
            "confidence": 0.91,
            "risk_flags": [
                "Follow_Up",
                "duplicate",
                "FOLLOW_UP",
                "Mixed Language",
                "ALL CAPS",
                "possible-template copy",
            ],
        }

    def chat_completion(self, *, model, messages, temperature, **kwargs):  # type: ignore[override]
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "kwargs": kwargs,
            }
        )
        payload = {
            "note": "Stub note",
            "analysis": self._analysis_payload,
        }
        serialized = json.dumps(payload, ensure_ascii=False)
        return {
            "mode": "content",
            "content_json": payload,
            "raw_content": serialized,
            "choices": [
                {
                    "message": {
                        "content": serialized,
                    }
                }
            ],
        }


def test_note_style_sender_sends_built_pack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    sid = "SID100"
    account_id = "idx-100"
    runs_root = tmp_path
    run_dir = runs_root / sid
    response_dir = run_dir / "frontend" / "review" / "responses"

    _write_manifest(run_dir, account_id)

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": "Please help, already paid."},
        },
    )

    monkeypatch.setattr(config, "NOTE_STYLE_ALLOW_TOOL_CALLS", True)
    monkeypatch.setattr(config, "NOTE_STYLE_RESPONSE_MODE", NoteStyleResponseMode.TOOL)
    monkeypatch.setattr(note_style_config, "NOTE_STYLE_ALLOW_TOOL_CALLS", True)
    monkeypatch.setattr(note_style_config, "NOTE_STYLE_RESPONSE_MODE", NoteStyleResponseMode.TOOL)

    build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)

    client = _StubClient()
    monkeypatch.setattr("backend.ai.note_style_sender.get_ai_client", lambda: client)

    caplog.set_level("INFO", logger="backend.ai.note_style_sender")

    processed = send_note_style_packs_for_sid(sid, runs_root=runs_root)
    assert processed == [account_id]
    assert len(client.calls) == 1
    first_kwargs = client.calls[0]["kwargs"]
    assert "tools" not in first_kwargs
    assert "tool_choice" not in first_kwargs
    assert first_kwargs.get("_note_style_request") is True

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)

    result_lines = [
        line
        for line in account_paths.result_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(result_lines) == 1
    stored_payload = json.loads(result_lines[0])
    required_keys = {
        "sid",
        "account_id",
        "evaluated_at",
        "analysis",
        "note_metrics",
    }
    assert required_keys.issubset(stored_payload.keys())
    note_hash_value = stored_payload.get("note_hash")
    if note_hash_value is not None:
        assert isinstance(note_hash_value, str)
        assert note_hash_value.strip()
    assert stored_payload["sid"] == sid
    assert stored_payload["account_id"] == account_id
    assert stored_payload["evaluated_at"].endswith("Z")

    pack_payload = json.loads(
        account_paths.pack_file.read_text(encoding="utf-8").splitlines()[0]
    )
    note_text = pack_payload["note_text"]
    assert stored_payload["note_metrics"] == {
        "char_len": len(note_text),
        "word_len": len(note_text.split()),
    }
    assert "prompt_salt" not in stored_payload
    assert "prompt_salt" not in pack_payload
    assert "fingerprint_hash" not in stored_payload
    assert "fingerprint_hash" not in pack_payload
    assert "fingerprint" not in stored_payload
    analysis = stored_payload["analysis"]
    assert set(analysis.keys()) == {
        "tone",
        "context_hints",
        "emphasis",
        "confidence",
        "risk_flags",
    }
    assert analysis["tone"] == "Empathetic"
    assert analysis["emphasis"] == ["paid_already", "custom", "support_request"]
    context = analysis["context_hints"]
    assert context["topic"] == "payment_dispute"
    timeframe = context["timeframe"]
    assert timeframe.get("relative") == "last_month"
    assert timeframe.get("month") in {None, "2024-04-01"}
    entities = context["entities"]
    assert entities["creditor"] == "capital one"
    assert entities["amount"] == pytest.approx(123.45)
    assert analysis["risk_flags"] == [
        "follow_up",
        "duplicate",
        "mixed_language",
        "all_caps",
        "possible_template_copy",
    ]

    note_metrics = stored_payload.get("note_metrics")
    assert isinstance(note_metrics, Mapping)
    assert note_metrics.get("char_len") > 0
    assert note_metrics.get("word_len") > 0
    assert set(note_metrics.keys()) == {"char_len", "word_len"}

    assert "account_context" not in stored_payload
    assert "bureaus_summary" not in stored_payload
    user_content = pack_payload["messages"][1]["content"]
    if isinstance(user_content, str):
        user_content = json.loads(user_content)
    assert user_content["note_text"]

    index_payload = json.loads(paths.index_file.read_text(encoding="utf-8"))
    packs = index_payload["packs"]
    assert packs[0]["status"] == "completed"
    assert "sent_at" not in packs[0]
    assert (
        packs[0]["result_path"]
        == account_paths.result_file.relative_to(paths.base).as_posix()
    )
    assert isinstance(packs[0].get("completed_at"), str)

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "backend.ai.note_style_sender"
    ]
    assert any("STYLE_SEND_ACCOUNT_START" in message for message in messages)
    assert any("STYLE_SEND_MODEL_CALL" in message for message in messages)
    assert any("STYLE_SEND_ACCOUNT_END" in message for message in messages)
    assert any("NOTE_STYLE_SENT" in message for message in messages)

    structured_records = [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == "backend.ai.note_style_sender"
        and record.getMessage().startswith("{")
    ]
    assert any(
        entry.get("event") == "NOTE_STYLE_SENT" and entry.get("account_id") == account_id
        for entry in structured_records
    )

    metrics_events = [
        entry
        for entry in structured_records
        if entry.get("event") == "NOTE_STYLE_MODEL_METRICS"
    ]
    assert metrics_events, "Expected model metrics event"
    metrics_entry = metrics_events[0]
    assert metrics_entry.get("parse_ok") is True
    assert metrics_entry.get("retry_count") == 0
    assert metrics_entry.get("model") == config.NOTE_STYLE_MODEL

    call_kwargs = client.calls[0]["kwargs"]
    assert "tools" not in call_kwargs


def test_note_style_sender_retries_on_invalid_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    sid = "SID110"
    account_id = "idx-110"
    runs_root = tmp_path
    run_dir = runs_root / sid
    response_dir = run_dir / "frontend" / "review" / "responses"

    _write_manifest(run_dir, account_id)

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": "Please process with retries."},
        },
    )

    monkeypatch.setattr(config, "NOTE_STYLE_ALLOW_TOOL_CALLS", True)
    monkeypatch.setattr(config, "NOTE_STYLE_RESPONSE_MODE", NoteStyleResponseMode.TOOL)
    monkeypatch.setattr(note_style_config, "NOTE_STYLE_ALLOW_TOOL_CALLS", True)
    monkeypatch.setattr(note_style_config, "NOTE_STYLE_RESPONSE_MODE", NoteStyleResponseMode.TOOL)

    build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)

    invalid_payload = {"choices": [{"message": {"content": "Not JSON"}}]}
    valid_analysis = {
        "analysis": {
            "tone": "formal",
            "context_hints": {
                "timeframe": {"month": "March", "relative": "last month"},
                "topic": "payment plan",
                "entities": {"creditor": "Example Bank", "amount": 125.5},
            },
            "emphasis": ["focus on empathy"],
            "confidence": 0.92,
            "risk_flags": ["compliance_check"],
        }
    }
    valid_payload_body = {"note": "Retry stub", **valid_analysis}
    valid_payload = {
        "mode": "content",
        "content_json": valid_payload_body,
        "raw_content": json.dumps(valid_payload_body, ensure_ascii=False),
        "choices": [
            {
                "message": {
                    "content": json.dumps(valid_payload_body, ensure_ascii=False)
                }
            }
        ],
    }

    class _RetryClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []
            self._responses = [invalid_payload, valid_payload]

        def chat_completion(self, *, model, messages, temperature, **kwargs):  # type: ignore[override]
            self.calls.append(
                {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "kwargs": kwargs,
                }
            )
            return self._responses[len(self.calls) - 1]

    client = _RetryClient()
    monkeypatch.setattr("backend.ai.note_style_sender.get_ai_client", lambda: client)
    monkeypatch.setattr(
        config,
        "NOTE_STYLE_INVALID_RESULT_RETRY_ATTEMPTS",
        2,
        raising=False,
    )
    monkeypatch.setattr(
        config,
        "NOTE_STYLE_INVALID_RESULT_RETRY_TOOL_CALL",
        False,
        raising=False,
    )

    caplog.set_level("INFO", logger="backend.ai.note_style_sender")

    processed = send_note_style_packs_for_sid(sid, runs_root=runs_root)

    assert processed == [account_id]
    assert len(client.calls) == 2
    first_kwargs = client.calls[0]["kwargs"]
    retry_kwargs = client.calls[1]["kwargs"]
    assert "tools" not in first_kwargs
    assert first_kwargs.get("response_format") == {"type": "json_object"}
    assert first_kwargs.get("_note_style_request") is True
    assert "tools" not in retry_kwargs
    assert retry_kwargs.get("response_format") == {"type": "json_object"}
    assert retry_kwargs.get("_note_style_request") is True

    corrective_message = client.calls[1]["messages"][1]
    assert corrective_message["role"] == "system"
    assert "previous output was not valid JSON" in corrective_message["content"]

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)

    raw_payload = account_paths.result_raw_file.read_text(encoding="utf-8").strip()
    assert raw_payload == "Not JSON"

    structured_records = [
        json.loads(record.getMessage())
        for record in caplog.records
        if record.name == "backend.ai.note_style_sender"
        and record.getMessage().startswith("{")
    ]

    retry_events = [
        entry
        for entry in structured_records
        if entry.get("event") == "NOTE_STYLE_INVALID_RESULT_RETRY"
    ]
    assert retry_events, "Expected structured retry event"
    assert retry_events[0]["attempt"] == 1

    sent_events = [
        entry
        for entry in structured_records
        if entry.get("event") == "NOTE_STYLE_SENT"
    ]
    assert sent_events and sent_events[0]["retries_used"] == 1

    metrics_events = [
        entry
        for entry in structured_records
        if entry.get("event") == "NOTE_STYLE_MODEL_METRICS"
    ]
    assert metrics_events, "Expected model metrics event on retry"
    metrics_entry = metrics_events[-1]
    assert metrics_entry.get("parse_ok") is True
    assert metrics_entry.get("retry_count") == 1



def test_note_style_sender_accepts_string_runs_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "SID200"
    account_id = "idx-200"
    runs_root = tmp_path
    run_dir = runs_root / sid
    response_dir = run_dir / "frontend" / "review" / "responses"

    _write_manifest(run_dir, account_id)

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": "The balance should be zero."},
        },
    )

    monkeypatch.setattr(config, "NOTE_STYLE_ALLOW_TOOL_CALLS", True)
    monkeypatch.setattr(config, "NOTE_STYLE_RESPONSE_MODE", NoteStyleResponseMode.TOOL)
    monkeypatch.setattr(note_style_config, "NOTE_STYLE_ALLOW_TOOL_CALLS", True)
    monkeypatch.setattr(note_style_config, "NOTE_STYLE_RESPONSE_MODE", NoteStyleResponseMode.TOOL)

    build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)

    client = _StubClient()
    monkeypatch.setattr("backend.ai.note_style_sender.get_ai_client", lambda: client)

    processed = send_note_style_packs_for_sid(sid, runs_root=os.fspath(runs_root))

    assert processed == [account_id]
    assert len(client.calls) == 1

def test_note_style_sender_skips_completed_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "SID101"
    account_id = "idx-101"
    runs_root = tmp_path
    run_dir = runs_root / sid
    response_dir = run_dir / "frontend" / "review" / "responses"

    _write_manifest(run_dir, account_id)

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {
                "explanation": "Please fix the errors on this account."
            },
        },
    )

    build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)

    client = _StubClient()
    monkeypatch.setattr("backend.ai.note_style_sender.get_ai_client", lambda: client)

    processed_first = send_note_style_packs_for_sid(sid, runs_root=runs_root)
    assert processed_first == [account_id]
    assert len(client.calls) == 1

    processed_second = send_note_style_packs_for_sid(sid, runs_root=runs_root)
    assert processed_second == []
    assert len(client.calls) == 1


def test_note_style_sender_skips_when_existing_result_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "SID200"
    account_id = "idx-200"
    runs_root = tmp_path
    run_dir = runs_root / sid
    response_dir = run_dir / "frontend" / "review" / "responses"

    _write_manifest(run_dir, account_id)

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": "Already fixed, thanks."},
        },
    )

    build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)

    pack_payload = json.loads(
        account_paths.pack_file.read_text(encoding="utf-8").splitlines()[0]
    )
    note_text = pack_payload["note_text"]
    final_result = {
        "sid": sid,
        "account_id": account_id,
        "note_metrics": {
            "char_len": len(note_text),
            "word_len": len(note_text.split()),
        },
        "analysis": {
            "tone": "neutral",
            "context_hints": {
                "timeframe": {"month": None, "relative": None},
                "topic": "other",
                "entities": {"creditor": None, "amount": None},
            },
            "emphasis": [],
            "confidence": 0.7,
            "risk_flags": ["follow_up"],
        },
    }
    store_note_style_result(
        sid,
        account_id,
        final_result,
        runs_root=runs_root,
        completed_at="2024-01-01T00:00:00Z",
    )
    complete_note_style_result(
        sid,
        account_id,
        runs_root=runs_root,
        completed_at="2024-01-01T00:00:00Z",
    )

    client = _StubClient()
    monkeypatch.setattr("backend.ai.note_style_sender.get_ai_client", lambda: client)

    processed = send_note_style_packs_for_sid(sid, runs_root=runs_root)

    assert processed == []
    assert client.calls == []

    updated_index = json.loads(paths.index_file.read_text(encoding="utf-8"))
    entry = updated_index["packs"][0]
    assert entry["status"] == "completed"
    assert "note_hash" not in entry
    assert entry.get("result_path") == account_paths.result_file.relative_to(paths.base).as_posix()


def test_note_style_sender_respects_skip_env_when_result_has_analysis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "SID300"
    account_id = "idx-300"
    runs_root = tmp_path
    run_dir = runs_root / sid
    response_dir = run_dir / "frontend" / "review" / "responses"

    _write_manifest(run_dir, account_id)

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": "Already reviewed."},
        },
    )

    build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)

    existing_result = {
        "sid": sid,
        "account_id": account_id,
        "analysis": {
            "tone": "calm",
            "context_hints": {
                "timeframe": {"month": None, "relative": None},
                "topic": "other",
                "entities": {"creditor": None, "amount": None},
            },
            "emphasis": [],
            "confidence": 0.5,
            "risk_flags": [],
        },
    }
    account_paths.result_file.write_text(
        json.dumps(existing_result, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    client = _StubClient()
    monkeypatch.setattr("backend.ai.note_style_sender.get_ai_client", lambda: client)
    monkeypatch.setattr(config, "NOTE_STYLE_SKIP_IF_RESULT_EXISTS", True)

    processed = send_note_style_packs_for_sid(sid, runs_root=runs_root)

    assert processed == []
    assert client.calls == []


def test_note_style_sender_respects_skip_env_when_result_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "SID301"
    account_id = "idx-301"
    runs_root = tmp_path
    run_dir = runs_root / sid
    response_dir = run_dir / "frontend" / "review" / "responses"

    _write_manifest(run_dir, account_id)

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": "Processing failed."},
        },
    )

    build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)

    failure_payload = {
        "status": "failed",
        "account": account_id,
        "sid": sid,
        "error": {"message": "upstream error"},
        "completed_at": "2024-01-01T00:00:00Z",
    }
    account_paths.result_file.write_text(
        json.dumps(failure_payload, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    client = _StubClient()
    monkeypatch.setattr("backend.ai.note_style_sender.get_ai_client", lambda: client)
    monkeypatch.setattr(config, "NOTE_STYLE_SKIP_IF_RESULT_EXISTS", True)

    processed = send_note_style_packs_for_sid(sid, runs_root=runs_root)

    assert processed == []
    assert client.calls == []


def test_note_style_sender_calls_when_skip_flag_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "SID305"
    account_id = "idx-305"
    runs_root = tmp_path
    run_dir = runs_root / sid
    response_dir = run_dir / "frontend" / "review" / "responses"

    _write_manifest(run_dir, account_id)

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": "Need info."},
        },
    )

    build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)

    existing_result = {
        "sid": sid,
        "account_id": account_id,
        "analysis": {
            "tone": "neutral",
            "context_hints": {
                "timeframe": {"month": None, "relative": None},
                "topic": "other",
                "entities": {"creditor": None, "amount": None},
            },
            "emphasis": [],
            "confidence": 0.5,
            "risk_flags": [],
        },
    }
    store_note_style_result(
        sid,
        account_id,
        existing_result,
        runs_root=runs_root,
        completed_at="2024-01-02T00:00:00Z",
    )
    complete_note_style_result(
        sid,
        account_id,
        runs_root=runs_root,
        completed_at="2024-01-02T00:00:00Z",
    )

    client = _StubClient()
    monkeypatch.setattr("backend.ai.note_style_sender.get_ai_client", lambda: client)
    monkeypatch.setattr(config, "NOTE_STYLE_SKIP_IF_RESULT_EXISTS", False)
    monkeypatch.setenv("NOTE_STYLE_SKIP_IF_RESULT_EXISTS", "0")

    processed = send_note_style_packs_for_sid(sid, runs_root=runs_root)

    assert processed == []
    assert client.calls == []

    stored_result = json.loads(
        account_paths.result_file.read_text(encoding="utf-8").splitlines()[0]
    )
    assert stored_result["analysis"]["tone"] == "neutral"


def test_note_style_sender_warns_but_sends_when_index_thin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sid = "SID306"
    account_id = "idx-306"
    runs_root = tmp_path
    run_dir = runs_root / sid
    response_dir = run_dir / "frontend" / "review" / "responses"

    _write_manifest(run_dir, account_id)

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": "Need review."},
        },
    )

    build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    # Overwrite the index with a very small payload so the warning triggers.
    paths.index_file.write_text("{}", encoding="utf-8")

    client = _StubClient()
    monkeypatch.setattr("backend.ai.note_style_sender.get_ai_client", lambda: client)
    monkeypatch.setattr(config, "NOTE_STYLE_WAIT_FOR_INDEX", True)
    monkeypatch.setenv("NOTE_STYLE_WAIT_FOR_INDEX", "1")

    caplog.set_level("WARNING", logger="backend.ai.note_style_sender")

    processed = send_note_style_packs_for_sid(sid, runs_root=runs_root)

    assert processed == [account_id]
    assert len(client.calls) == 1

    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "backend.ai.note_style_sender"
    ]
    assert any("NOTE_STYLE_INDEX_THIN" in message for message in messages)


def test_note_style_sender_raises_when_pack_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "SID102"
    account_id = "idx-102"
    runs_root = tmp_path
    run_dir = runs_root / sid
    response_dir = run_dir / "frontend" / "review" / "responses"

    _write_manifest(run_dir, account_id)

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": "Need support"},
        },
    )

    build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)
    account_paths.pack_file.unlink()

    client = _StubClient()
    monkeypatch.setattr("backend.ai.note_style_sender.get_ai_client", lambda: client)
    processed = send_note_style_packs_for_sid(sid, runs_root=runs_root)
    assert processed == []
    assert len(client.calls) == 0


def test_note_style_sender_strips_debug_message_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "SID300"
    account_id = "idx-300"
    runs_root = tmp_path
    run_dir = runs_root / sid
    response_dir = run_dir / "frontend" / "review" / "responses"

    _write_manifest(run_dir, account_id)

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": "Need info."},
        },
    )

    build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)

    payload = json.loads(account_paths.pack_file.read_text(encoding="utf-8").splitlines()[0])
    payload["messages"][1]["debug_snapshot"] = {"should": "not-travel"}
    payload["messages"][1]["raw_payload"] = {"secret": True}
    account_paths.pack_file.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    client = _StubClient()
    monkeypatch.setattr("backend.ai.note_style_sender.get_ai_client", lambda: client)

    processed = send_note_style_packs_for_sid(sid, runs_root=runs_root)

    assert processed == [account_id]
    assert len(client.calls) == 1

    call_messages = client.calls[0]["messages"]
    assert isinstance(call_messages, list)
    user_entry = call_messages[1]
    assert "debug_snapshot" not in user_entry
    assert "raw_payload" not in user_entry


def test_note_style_sender_normalizes_message_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sid = "SID320"
    account_id = "idx-320"
    runs_root = tmp_path
    run_dir = runs_root / sid
    response_dir = run_dir / "frontend" / "review" / "responses"

    _write_manifest(run_dir, account_id)

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": "Need support."},
        },
    )

    monkeypatch.setattr(config, "NOTE_STYLE_ALLOW_TOOL_CALLS", True)
    monkeypatch.setattr(config, "NOTE_STYLE_RESPONSE_MODE", NoteStyleResponseMode.TOOL)
    monkeypatch.setattr(note_style_config, "NOTE_STYLE_ALLOW_TOOL_CALLS", True)
    monkeypatch.setattr(note_style_config, "NOTE_STYLE_RESPONSE_MODE", NoteStyleResponseMode.TOOL)

    build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)

    pack_payload = json.loads(account_paths.pack_file.read_text(encoding="utf-8").splitlines()[0])
    pack_payload["messages"][0]["content"] = [
        {"type": "text", "text": "System override"},
        {"type": "text", "text": "Second"},
    ]
    pack_payload["messages"][1]["content"] = {"payload": {"topic": "billing"}}
    pack_payload["response_format"] = "json_object"
    account_paths.pack_file.write_text(
        json.dumps(pack_payload, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    client = _StubClient()
    monkeypatch.setattr("backend.ai.note_style_sender.get_ai_client", lambda: client)

    processed = send_note_style_packs_for_sid(sid, runs_root=runs_root)

    assert processed == [account_id]
    assert len(client.calls) == 1

    call_messages = client.calls[0]["messages"]
    assert isinstance(call_messages, list)
    system_content = call_messages[0]["content"]
    assert isinstance(system_content, str)
    assert "exactly ONE JSON object" in system_content
    user_content = call_messages[1]["content"]
    assert isinstance(user_content, str)
    assert json.loads(user_content) == {"payload": {"topic": "billing"}}

    call_kwargs = client.calls[0]["kwargs"]
    assert "tools" not in call_kwargs
    assert call_kwargs.get("response_format") == {"type": "json_object"}
    assert call_kwargs.get("_note_style_request") is True


def test_note_style_sender_ignores_debug_snapshot_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "SID301"
    account_id = "idx-301"
    runs_root = tmp_path
    run_dir = runs_root / sid
    response_dir = run_dir / "frontend" / "review" / "responses"

    _write_manifest(run_dir, account_id)

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": "Need info."},
        },
    )

    build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    debug_candidate = paths.debug_dir / "acc_debug.jsonl"
    debug_candidate.parent.mkdir(parents=True, exist_ok=True)
    debug_candidate.write_text(
        json.dumps({
            "messages": [
                {"role": "system", "content": "bad"},
                {"role": "user", "content": "bad"},
            ]
        }, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    client = _StubClient()
    monkeypatch.setattr("backend.ai.note_style_sender.get_ai_client", lambda: client)
    monkeypatch.setenv("NOTE_STYLE_PACK_GLOB", "**/*.jsonl")

    processed = send_note_style_packs_for_sid(sid, runs_root=runs_root)

    assert processed == [account_id]
    assert len(client.calls) == 1


def test_note_style_sender_uses_manifest_pack_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "SID410"
    account_id = "idx-410"
    runs_root = tmp_path
    run_dir = runs_root / sid
    response_dir = run_dir / "frontend" / "review" / "responses"

    _write_manifest(run_dir, account_id)

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": "Please escalate."},
        },
    )

    build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)

    index_payload = json.loads(paths.index_file.read_text(encoding="utf-8"))
    entry = index_payload["packs"][0]
    entry["pack_path"] = (
        f"C:\\author\\runs\\{sid}\\ai_packs\\note_style\\packs\\{account_paths.pack_file.name}"
    )
    entry["result_path"] = (
        f"C:\\author\\runs\\{sid}\\ai_packs\\note_style\\results\\{account_paths.result_file.name}"
    )
    paths.index_file.write_text(
        json.dumps(index_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(config, "NOTE_STYLE_USE_MANIFEST_PATHS", True, raising=False)

    client = _StubClient()
    monkeypatch.setattr("backend.ai.note_style_sender.get_ai_client", lambda: client)

    processed = send_note_style_packs_for_sid(sid, runs_root=runs_root)

    assert processed == [account_id]
    assert len(client.calls) == 1

    result_path = account_paths.result_file
    assert result_path.exists()

    updated_index = json.loads(paths.index_file.read_text(encoding="utf-8"))
    updated_entry = updated_index["packs"][0]
    assert (
        updated_entry.get("pack")
        == account_paths.pack_file.relative_to(paths.base).as_posix()
    )
    assert (
        updated_entry["result_path"]
        == result_path.relative_to(paths.base).as_posix()
    )


def test_note_style_pack_for_account_uses_manifest_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "SID411"
    account_id = "idx-411"
    runs_root = tmp_path
    run_dir = runs_root / sid
    response_dir = run_dir / "frontend" / "review" / "responses"

    _write_manifest(run_dir, account_id)

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": "Handle directly."},
        },
    )

    build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)

    index_payload = json.loads(paths.index_file.read_text(encoding="utf-8"))
    entry = index_payload["packs"][0]
    entry["pack_path"] = (
        f"C:\\author\\runs\\{sid}\\ai_packs\\note_style\\packs\\{account_paths.pack_file.name}"
    )
    entry["result_path"] = (
        f"C:\\author\\runs\\{sid}\\ai_packs\\note_style\\results\\{account_paths.result_file.name}"
    )
    paths.index_file.write_text(
        json.dumps(index_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(config, "NOTE_STYLE_USE_MANIFEST_PATHS", True, raising=False)

    client = _StubClient()
    monkeypatch.setattr("backend.ai.note_style_sender.get_ai_client", lambda: client)

    processed = send_note_style_pack_for_account(
        sid, account_id, runs_root=runs_root
    )

    assert processed is True
    assert len(client.calls) == 1

    result_path = account_paths.result_file
    assert result_path.exists()

    updated_index = json.loads(paths.index_file.read_text(encoding="utf-8"))
    updated_entry = updated_index["packs"][0]
    assert (
        updated_entry.get("pack")
        == account_paths.pack_file.relative_to(paths.base).as_posix()
    )
    assert (
        updated_entry["result_path"]
        == result_path.relative_to(paths.base).as_posix()
    )



def test_note_style_sender_enforces_json_mode_even_when_tool_env_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sid = "SID120"
    account_id = "idx-120"
    runs_root = tmp_path
    run_dir = runs_root / sid
    response_dir = run_dir / "frontend" / "review" / "responses"

    _write_manifest(run_dir, account_id)

    _write_response(
        response_dir / f"{account_id}.result.json",
        {
            "sid": sid,
            "account_id": account_id,
            "answers": {"explanation": "Tool mode please."},
        },
    )

    monkeypatch.setattr(config, "NOTE_STYLE_ALLOW_TOOL_CALLS", True)
    monkeypatch.setattr(config, "NOTE_STYLE_ALLOW_TOOLS", True)
    monkeypatch.setattr(config, "NOTE_STYLE_RESPONSE_MODE", NoteStyleResponseMode.TOOL)
    monkeypatch.setattr(note_style_config, "NOTE_STYLE_ALLOW_TOOL_CALLS", True)
    monkeypatch.setattr(note_style_config, "NOTE_STYLE_ALLOW_TOOLS", True)
    monkeypatch.setattr(
        note_style_config, "NOTE_STYLE_RESPONSE_MODE", NoteStyleResponseMode.TOOL
    )

    build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)

    analysis_body = {
        "note": "Structured analysis",
        "analysis": {
            "tone": "empathetic",
            "context_hints": {
                "timeframe": {"month": "2024-04", "relative": "this year"},
                "topic": "repayment plan",
                "entities": {"creditor": "Example Bank", "amount": 451.0},
            },
            "emphasis": ["payment_plan"],
            "confidence": 0.8,
            "risk_flags": ["follow_up"],
        },
    }

    tool_payload = {
        "mode": "tool",
        "tool_json": analysis_body,
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "submit_note_style_analysis",
                                "arguments": json.dumps(analysis_body, ensure_ascii=False),
                            },
                        }
                    ]
                }
            }
        ],
    }

    class _ToolClient:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def chat_completion(self, *, model, messages, temperature, **kwargs):  # type: ignore[override]
            self.calls.append(
                {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "kwargs": kwargs,
                }
            )
            return tool_payload

    client = _ToolClient()
    monkeypatch.setattr("backend.ai.note_style_sender.get_ai_client", lambda: client)

    processed = send_note_style_packs_for_sid(sid, runs_root=runs_root)
    assert processed == [account_id]
    assert len(client.calls) == 1
    call_kwargs = client.calls[0]["kwargs"]
    assert "tools" not in call_kwargs
    assert call_kwargs.get("response_format") == {"type": "json_object"}

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)

    result_lines = [
        line
        for line in account_paths.result_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(result_lines) == 1
    stored_payload = json.loads(result_lines[0])
    assert stored_payload["analysis"]["tone"] == "empathetic"

