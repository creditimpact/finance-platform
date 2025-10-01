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


class _SmokeStubClient:
    def create(self, *, model: str, messages, response_format):  # type: ignore[override]
        payload = {
            "decision": "strong",
            "rationale": "auto",
            "citations": [],
            "confidence": 0.75,
        }
        return {"choices": [{"message": {"content": json.dumps(payload)}}]}


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


def test_validation_sender_smoke_writes_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sid = "SMOKE001"
    base_dir = tmp_path / "runs" / sid / "ai_packs" / "validation"
    packs_dir = base_dir / "packs"
    results_dir = base_dir / "results"
    index_path = base_dir / "index.json"

    packs_dir.mkdir(parents=True, exist_ok=True)

    pack1 = packs_dir / "account_001.pack.jsonl"
    pack1.write_text("\n".join((_pack_line("Account Name"), "")), encoding="utf-8")

    pack2 = packs_dir / "account_002.pack.jsonl"
    pack2.write_text("\n".join((_pack_line("Balance"), "")), encoding="utf-8")

    manifest = {
        "schema_version": 2,
        "sid": sid,
        "root": ".",
        "packs_dir": "packs",
        "results_dir": "results",
        "packs": [
            {
                "account_id": 1,
                "pack": "packs/account_001.pack.jsonl",
                "result_jsonl": "results/account_001.result.jsonl",
                "result_json": "results/account_001.result.json",
                "lines": 1,
                "status": "built",
                "built_at": "2024-01-01T00:00:00Z",
            },
            {
                "account_id": 2,
                "pack": "packs/account_002.pack.jsonl",
                "result_jsonl": "results/account_002.result.jsonl",
                "result_json": "results/account_002.result.json",
                "lines": 1,
                "status": "built",
                "built_at": "2024-01-01T00:00:00Z",
            },
        ],
    }

    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    monkeypatch.setenv("AI_MODEL", "stub-model")

    sender = ValidationPackSender(index_path, http_client=_SmokeStubClient())
    results = sender.send()

    assert {item["account_id"] for item in results} == {1, 2}

    for account_id in (1, 2):
        jsonl_file = results_dir / f"account_{account_id:03d}.result.jsonl"
        summary_file = results_dir / f"account_{account_id:03d}.result.json"

        assert jsonl_file.is_file()
        assert summary_file.is_file()

        lines = [line for line in jsonl_file.read_text(encoding="utf-8").splitlines() if line]
        assert lines, f"expected result lines for account {account_id}"

        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        assert summary["status"] == "done"
        assert summary["account_id"] == account_id
        assert summary["results"], "summary should include results"
