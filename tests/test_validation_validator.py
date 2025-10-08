from typing import Any, Mapping

from backend.validation.schema import validate_llm_decision


def _base_decision() -> dict[str, Any]:
    return {
        "sid": "S123",
        "account_id": 1,
        "id": "line-1",
        "field": "account_type",
        "decision": "strong",
        "rationale": "C4_TWO_MATCH_ONE_DIFF supports the consumer (C4_TWO_MATCH_ONE_DIFF).",
        "citations": ["equifax: revolving"],
        "reason_code": "C4_TWO_MATCH_ONE_DIFF",
        "reason_label": "Account type mismatch",
        "modifiers": {
            "material_mismatch": True,
            "time_anchor": False,
            "doc_dependency": False,
        },
        "confidence": 0.9,
    }


def _finding() -> Mapping[str, Any]:
    return {
        "bureaus": {
            "equifax": {"normalized": "revolving", "raw": "Revolving"},
            "experian": {"normalized": "installment", "raw": "Installment"},
        },
        "documents": ["statement"],
    }


def test_validate_llm_decision_rejects_empty_citations() -> None:
    decision = _base_decision()
    decision["citations"] = []

    ok, errors = validate_llm_decision(decision, _finding())

    assert not ok
    assert "non-empty" in errors[0]


def test_validate_llm_decision_requires_reason_code_in_rationale() -> None:
    decision = _base_decision()
    decision["rationale"] = "Mismatch supports the consumer."

    ok, errors = validate_llm_decision(decision, _finding())

    assert not ok
    assert "rationale_missing_reason_code" in errors


def test_validate_llm_decision_accepts_valid_payload() -> None:
    decision = _base_decision()
    decision["citations"] = ["equifax: revolving", "experian: installment"]

    ok, errors = validate_llm_decision(decision, _finding())

    assert ok
    assert errors == []


