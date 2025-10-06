import json
from pathlib import Path

from backend.ai.validation_builder import ValidationPackWriter
from backend.ai.validation_results import store_validation_result
from backend.core.ai.paths import (
    validation_pack_filename_for_account,
    validation_packs_dir,
    validation_result_jsonl_filename_for_account,
    validation_result_summary_filename_for_account,
    validation_results_dir,
)
from backend.validation.send_packs import ValidationPackSender


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_summary(*requirements: dict[str, object]) -> dict[str, object]:
    return {
        "validation_requirements": {
            "findings": [dict(req) for req in requirements],
            "field_consistency": {},
        }
    }


class _StubClient:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload
        self.requests: list[dict[str, object]] = []

    def create(
        self,
        *,
        model: str,
        messages,
        response_format,
        pack_id=None,
        on_error=None,
    ):  # type: ignore[override]
        self.requests.append(
            {
                "model": model,
                "messages": messages,
                "response_format": response_format,
                "pack_id": pack_id,
            }
        )
        return {"choices": [{"message": {"content": json.dumps(self._payload)}}]}


def test_single_result_summary_only(tmp_path: Path) -> None:
    sid = "SID500"
    account_id = 5
    runs_root = tmp_path / "runs"

    finding = {
        "field": "account_type",
        "is_mismatch": True,
        "ai_needed": True,
        "send_to_ai": True,
        "bureau_values": {
            "transunion": {"raw": "installment"},
            "experian": {"raw": "installment"},
            "equifax": {"raw": "installment"},
        },
    }

    account_dir = runs_root / sid / "cases" / "accounts" / str(account_id)
    _write_json(account_dir / "summary.json", _build_summary(finding))

    writer = ValidationPackWriter(sid, runs_root=runs_root)
    lines = writer.write_pack_for_account(account_id)

    assert len(lines) == 1
    field_id = str(lines[0].payload["id"])

    pack_path = (
        validation_packs_dir(sid, runs_root=runs_root)
        / validation_pack_filename_for_account(account_id)
    )
    assert pack_path.exists()

    response_payload = {
        "decision_per_field": [
            {
                "id": field_id,
                "decision": "strong",
                "rationale": "Mismatch supports the consumer",
                "citations": ["equifax: installment"],
            }
        ]
    }

    summary_path = store_validation_result(
        sid,
        account_id,
        response_payload,
        runs_root=runs_root,
        status="done",
    )

    results_dir = validation_results_dir(sid, runs_root=runs_root)
    summary_file = results_dir / validation_result_summary_filename_for_account(account_id)
    jsonl_file = results_dir / validation_result_jsonl_filename_for_account(account_id)

    assert summary_path == summary_file
    assert summary_file.exists()
    assert not jsonl_file.exists()

    summary_payload = json.loads(summary_file.read_text(encoding="utf-8"))
    assert summary_payload["decisions"] == [
        {
            "field_id": field_id,
            "decision": "strong",
            "rationale": "Mismatch supports the consumer",
            "citations": ["equifax: installment"],
        }
    ]


def test_sender_accepts_valid_json_response(tmp_path: Path) -> None:
    sid = "SID501"
    manifest_path = tmp_path / "index.json"

    manifest = {
        "schema_version": 2,
        "sid": sid,
        "root": ".",
        "packs_dir": "packs",
        "results_dir": "results",
        "packs": [],
        "__index_path__": manifest_path,
    }

    payload = {
        "decision": "strong",
        "justification": "Values diverge in the consumer's favor.",
        "labels": ["policy_match"],
        "citations": ["transunion.normalized"],
        "confidence": 0.82,
    }

    client = _StubClient(payload)
    sender = ValidationPackSender(manifest, http_client=client)

    pack_line = {
        "prompt": {
            "system": "system guidance",
            "user": {"field": "account_type", "context": "value"},
        }
    }

    response = sender._call_model(
        pack_line,
        account_id=1,
        account_label="001",
        line_number=1,
        line_id="acc_001__account_type",
        pack_id="acc_001",
        error_path=tmp_path / "acc_001.result.error.json",
    )

    assert response == payload
    assert client.requests
    last_request = client.requests[-1]
    assert last_request["response_format"] == {"type": "json_object"}
    assert last_request["model"] == sender.model

    serialized_user = json.dumps(pack_line["prompt"]["user"], ensure_ascii=False, sort_keys=True)
    assert last_request["messages"][1]["content"] == serialized_user
