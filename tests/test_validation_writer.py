import json
from pathlib import Path
from typing import Any, Callable, Mapping

import pytest

from backend.pipeline.runs import _utc_now
from backend.validation.index_schema import ValidationIndex, ValidationPackRecord
from backend.validation.send_packs import (
    ValidationPackError,
    ValidationPackSender,
    _ChatCompletionResponse,
    _ManifestView,
)


def _build_sender(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ValidationPackSender, Path, Path, ValidationPackRecord]:
    base = tmp_path / "runs" / "S123" / "ai_packs" / "validation"
    packs_dir = base / "packs"
    results_dir = base / "results"
    packs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    index_path = base / "index.json"
    record = ValidationPackRecord(
        account_id=1,
        pack="packs/val_acc_001.jsonl",
        result_jsonl="results/acc_001.result.jsonl",
        result_json="results/acc_001.result.json",
        lines=1,
        status="built",
        built_at=_utc_now(),
    )
    index = ValidationIndex(
        index_path=index_path,
        sid="S123",
        packs_dir="packs",
        results_dir="results",
        packs=(record,),
    )

    view = _ManifestView(index=index, log_path=base / "logs.txt")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    sender = ValidationPackSender(index_path, preloaded_view=view)
    return sender, packs_dir, results_dir, record


def _write_pack_line(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_writer_creates_single_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sender, _packs_dir, results_dir, _record = _build_sender(tmp_path, monkeypatch)

    lines = [
        {
            "id": "acc_001__field_001",
            "account_id": 1,
            "field": "account_type",
            "decision": "strong",
            "rationale": "C4_TWO_MATCH_ONE_DIFF supports the consumer",
            "citations": ["equifax: revolving"],
        },
        {
            "id": "acc_001__field_002",
            "account_id": 1,
            "field": "balance_owed",
            "decision": "supportive",
            "rationale": "C5_ALL_DIFF shows mismatch across bureaus",
            "citations": ["equifax: 100", "experian: 200"],
        },
    ]

    jsonl_path, summary_path = sender._write_results(1, lines)

    assert jsonl_path == summary_path
    assert jsonl_path.name == "acc_001.result.jsonl"
    assert not (results_dir / "acc_001.result.json").exists()

    contents = jsonl_path.read_text(encoding="utf-8").splitlines()
    assert len(contents) == 2
    assert json.loads(contents[0])["field"] == "account_type"
    assert json.loads(contents[1])["field"] == "balance_owed"


def test_send_to_ai_false_uses_deterministic_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sender, packs_dir, results_dir, record = _build_sender(tmp_path, monkeypatch)

    pack_line = {
        "id": "line-1",
        "field": "account_type",
        "sid": "S123",
        "reason_code": "C4_TWO_MATCH_ONE_DIFF",
        "reason_label": "Account type mismatch",
        "send_to_ai": False,
        "bureaus": {
            "equifax": {"normalized": "revolving", "raw": "Revolving"},
            "experian": {"normalized": "revolving", "raw": "Revolving"},
            "transunion": {"normalized": "installment", "raw": "Installment"},
        },
        "finding": {
            "is_mismatch": True,
            "bureaus": {
                "equifax": {"normalized": "revolving", "raw": "Revolving"},
                "experian": {"normalized": "revolving", "raw": "Revolving"},
                "transunion": {"normalized": "installment", "raw": "Installment"},
            },
            "documents": ["statement"],
        },
    }
    pack_path = packs_dir / "val_acc_001.jsonl"
    _write_pack_line(pack_path, pack_line)

    def _fail_call(*_args: Any, **_kwargs: Any) -> None:
        raise ValidationPackError("model should not be called")

    sender._call_model = _fail_call  # type: ignore[assignment]

    result = sender._process_account(
        record.account_id,
        record.account_id,
        pack_path,
        record.pack,
        results_dir / "acc_001.result.jsonl",
        record.result_jsonl,
        results_dir / "acc_001.result.json",
        record.result_json,
    )

    assert result["results"]
    decision = result["results"][0]
    assert decision["decision"] in {"supportive", "neutral", "no_case", "strong"}
    assert "citations" in decision
    assert not (results_dir / "acc_001.result.json").exists()


class _StubClient:
    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = responses
        self.calls: list[dict[str, Any]] = []

    def create(
        self,
        payload: Mapping[str, Any],
        *,
        pack_id: str | None = None,
        on_error: Callable[[int, str], None] | None = None,
    ) -> _ChatCompletionResponse:
        self.calls.append(dict(payload))
        response_payload = self._responses.pop(0)
        return _ChatCompletionResponse(
            payload=response_payload,
            status_code=200,
            latency=0.01,
            retries=0,
        )


def test_call_model_retries_with_correction_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sender, _packs_dir, results_dir, _record = _build_sender(tmp_path, monkeypatch)

    pack_line = {
        "id": "line-1",
        "field": "account_type",
        "sid": "S123",
        "reason_code": "C4_TWO_MATCH_ONE_DIFF",
        "reason_label": "Account type mismatch",
        "finding": {
            "bureaus": {
                "equifax": {"normalized": "revolving", "raw": "Revolving"},
                "experian": {"normalized": "installment", "raw": "Installment"},
            },
            "documents": ["statement"],
        },
    }

    invalid_response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "sid": "S123",
                            "account_id": 1,
                            "id": "line-1",
                            "field": "account_type",
                            "decision": "strong",
                            "rationale": "Account type mismatch",
                            "citations": [],
                            "reason_code": "C4_TWO_MATCH_ONE_DIFF",
                            "reason_label": "Account type mismatch",
                            "modifiers": {
                                "material_mismatch": True,
                                "time_anchor": False,
                                "doc_dependency": False,
                            },
                            "confidence": 0.8,
                        }
                    )
                }
            }
        ]
    }

    valid_response = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "sid": "S123",
                            "account_id": 1,
                            "id": "line-1",
                            "field": "account_type",
                            "decision": "strong",
                            "rationale": "Account type mismatch (C4_TWO_MATCH_ONE_DIFF)",
                            "citations": ["equifax: revolving"],
                            "reason_code": "C4_TWO_MATCH_ONE_DIFF",
                            "reason_label": "Account type mismatch",
                            "modifiers": {
                                "material_mismatch": True,
                                "time_anchor": False,
                                "doc_dependency": False,
                            },
                            "confidence": 0.82,
                        }
                    )
                }
            }
        ]
    }

    stub_client = _StubClient([invalid_response, valid_response])
    sender._client = stub_client  # type: ignore[assignment]

    result = sender._call_model(
        pack_line,
        account_id=1,
        account_label="001",
        line_number=1,
        line_id="line-1",
        pack_id="acc_001",
        error_path=results_dir / "acc_001.result.error.json",
        result_path=results_dir / "acc_001.result.jsonl",
        result_display="results/acc_001.result.jsonl",
    )

    assert len(stub_client.calls) == 2
    retry_prompt = stub_client.calls[1]["messages"][1]["content"]
    assert "FIX:" in retry_prompt
    assert "invalid" in retry_prompt
    assert result["citations"] == ["equifax: revolving"]
    assert result["decision"] == "strong"


