import json
import sys
import types
from pathlib import Path

import pytest

from backend.validation.send_packs import ValidationPackSender


_requests_stub = types.ModuleType("requests")


def _post_stub(*args, **kwargs):  # pragma: no cover - safety net
    raise AssertionError("requests.post should not be invoked in tests")


_requests_stub.post = _post_stub
sys.modules.setdefault("requests", _requests_stub)


class _FixedResponseStubClient:
    def __init__(self, payload: dict[str, object]):
        self._payload = payload

    def create(self, *, model: str, messages, response_format):  # type: ignore[override]
        return {"choices": [{"message": {"content": json.dumps(self._payload)}}]}


def _pack_line(field: str) -> str:
    payload = {
        "id": field,
        "field": field,
        "prompt": {
            "system": "system",
            "user": {"field": field, "context": "value"},
        },
    }
    return json.dumps(payload)


def _seed_manifest(
    tmp_path: Path, account_ids: list[int], sid: str = "SMOKE001"
) -> tuple[str, Path, Path]:
    base_dir = tmp_path / "runs" / sid / "ai_packs" / "validation"
    packs_dir = base_dir / "packs"
    results_dir = base_dir / "results"
    index_path = base_dir / "index.json"

    packs_dir.mkdir(parents=True, exist_ok=True)

    manifest_accounts: list[dict[str, object]] = []
    for account_id in account_ids:
        pack_path = packs_dir / f"account_{account_id:03d}.pack.jsonl"
        pack_path.write_text(
            "\n".join((_pack_line(f"Field {account_id}"), "")),
            encoding="utf-8",
        )
        manifest_accounts.append(
            {
                "account_id": account_id,
                "pack": f"packs/account_{account_id:03d}.pack.jsonl",
                "result_jsonl": f"results/account_{account_id:03d}.result.jsonl",
                "result_json": f"results/account_{account_id:03d}.result.json",
                "lines": 1,
                "status": "built",
                "built_at": "2024-01-01T00:00:00Z",
            }
        )

    manifest = {
        "schema_version": 2,
        "sid": sid,
        "root": ".",
        "packs_dir": "packs",
        "results_dir": "results",
        "packs": manifest_accounts,
    }

    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return sid, index_path, results_dir


def test_validation_sender_smoke_writes_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid, index_path, results_dir = _seed_manifest(tmp_path, [1, 2])
    monkeypatch.setenv("AI_MODEL", "stub-model")
    monkeypatch.setenv("VALIDATION_SINGLE_RESULT_FILE", "0")

    payload = {
        "decision": "strong",
        "justification": "auto",
        "labels": ["deterministic_match"],
        "citations": ["transunion.raw"],
        "confidence": 0.75,
    }

    sender = ValidationPackSender(
        index_path, http_client=_FixedResponseStubClient(payload)
    )
    results = sender.send()

    assert {item["account_id"] for item in results} == {1, 2}

    for account_id in (1, 2):
        jsonl_file = results_dir / f"account_{account_id:03d}.result.jsonl"
        summary_file = results_dir / f"account_{account_id:03d}.result.json"

        assert jsonl_file.is_file()
        assert summary_file.is_file()

        lines = [line for line in jsonl_file.read_text(encoding="utf-8").splitlines() if line]
        assert lines, f"expected result lines for account {account_id}"
        entry = json.loads(lines[0])
        assert entry["decision"] == "strong"
        assert entry["labels"] == ["deterministic_match"]
        assert entry["citations"] == ["transunion.raw"]
        assert entry["confidence"] == 0.75

        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        assert summary["status"] == "done"
        assert summary["account_id"] == account_id
        assert summary["results"], "summary should include results"
        summary_entry = summary["results"][0]
        assert summary_entry["labels"] == ["deterministic_match"]


def test_validation_sender_invalid_response_guardrail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, index_path, results_dir = _seed_manifest(tmp_path, [1], sid="GUARD001")
    monkeypatch.setenv("AI_MODEL", "stub-model")
    monkeypatch.setenv("ENABLE_OBSERVABILITY_H", "1")
    monkeypatch.setenv("VALIDATION_SINGLE_RESULT_FILE", "0")

    from backend.analytics import analytics_tracker

    analytics_tracker.reset_counters()

    payload = {
        "decision": "strong",
        "justification": "auto",
        "citations": ["transunion.raw"],
        "confidence": 0.92,
    }

    sender = ValidationPackSender(
        index_path, http_client=_FixedResponseStubClient(payload)
    )
    sender.send()

    summary_path = results_dir / "account_001.result.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result_entry = summary["results"][0]

    assert result_entry["decision"] == "no_case"
    assert result_entry["rationale"] == "[guardrail:invalid_response]"
    assert "labels" not in result_entry

    counters = analytics_tracker.get_counters()
    assert counters.get("validation.ai.response_invalid", 0) >= 1


def test_validation_sender_low_confidence_guardrail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, index_path, results_dir = _seed_manifest(tmp_path, [1], sid="GUARD002")
    monkeypatch.setenv("AI_MODEL", "stub-model")
    monkeypatch.setenv("ENABLE_OBSERVABILITY_H", "1")
    monkeypatch.setenv("VALIDATION_SINGLE_RESULT_FILE", "0")

    from backend.analytics import analytics_tracker

    analytics_tracker.reset_counters()

    payload = {
        "decision": "strong",
        "justification": "auto",
        "labels": ["semantic_review"],
        "citations": ["equifax.normalized"],
        "confidence": 0.25,
    }

    sender = ValidationPackSender(
        index_path, http_client=_FixedResponseStubClient(payload)
    )
    sender.send()

    summary_path = results_dir / "account_001.result.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result_entry = summary["results"][0]

    assert result_entry["decision"] == "no_case"
    assert result_entry["rationale"].endswith("[guardrail:low_confidence]")
    assert result_entry["labels"] == ["semantic_review"]
    assert result_entry["confidence"] == 0.25

    counters = analytics_tracker.get_counters()
    assert counters.get("validation.ai.response_low_confidence", 0) >= 1
