import sys
from types import SimpleNamespace
from typing import Optional

if "requests" not in sys.modules:
    sys.modules["requests"] = SimpleNamespace(post=lambda *args, **kwargs: None)

from backend.core.logic.validation_field_sets import (
    ALL_VALIDATION_FIELDS,
    ALWAYS_INVESTIGATABLE_FIELDS,
    CONDITIONAL_FIELDS,
)
from backend.validation.send_packs import _CONDITIONAL_FIELDS, _ALLOWED_FIELDS
from backend.validation.send_packs import _ALWAYS_INVESTIGATABLE_FIELDS
from backend.validation.send_packs import _enforce_conditional_gate
from backend.ai import validation_builder


def _account_number_pack_line(last4_a: str, last4_b: Optional[str]) -> dict:
    bureaus = {
        "transunion": {
            "raw": f"****{last4_a}",
            "normalized": {"display": f"****{last4_a}", "last4": last4_a},
        },
        "experian": {
            "raw": f"{last4_b}" if last4_b else None,
            "normalized": (
                {"display": f"{last4_b}", "last4": last4_b}
                if last4_b
                else {"display": "", "last4": None}
            ),
        },
        "equifax": {"raw": None, "normalized": {"display": "", "last4": None}},
    }

    return {
        "field": "account_number_display",
        "conditional_gate": True,
        "min_corroboration": 2,
        "bureaus": bureaus,
    }


def test_account_number_gate_rejects_masking_only() -> None:
    decision, rationale, info = _enforce_conditional_gate(
        "account_number_display",
        "strong",
        "Masking only",
        _account_number_pack_line("1234", "1234"),
    )

    assert decision == "no_case"
    assert "conditional_gate" in rationale
    assert info is not None
    assert info["reason"] == "insufficient_evidence"


def test_validation_field_sets_match_spec() -> None:
    expected_always = tuple(ALWAYS_INVESTIGATABLE_FIELDS)
    expected_conditional = tuple(CONDITIONAL_FIELDS)
    expected_all = set(ALL_VALIDATION_FIELDS)

    assert _ALWAYS_INVESTIGATABLE_FIELDS == expected_always
    assert _CONDITIONAL_FIELDS == expected_conditional
    assert _ALLOWED_FIELDS == expected_all

    assert set(validation_builder._ALWAYS_INVESTIGATABLE_FIELDS) == set(
        expected_always
    )
    assert set(validation_builder._CONDITIONAL_FIELDS) == set(expected_conditional)
    assert validation_builder._ALLOWED_FIELDS == expected_all


def test_account_number_gate_allows_true_conflict() -> None:
    decision, rationale, info = _enforce_conditional_gate(
        "account_number_display",
        "strong",
        "Digits conflict",
        _account_number_pack_line("1234", "5678"),
    )

    assert decision == "strong"
    assert rationale == "Digits conflict"
    assert info is None


def test_account_rating_gate_needs_multiple_values() -> None:
    pack_line = {
        "field": "account_rating",
        "conditional_gate": True,
        "min_corroboration": 2,
        "bureaus": {
            "transunion": {"raw": "A", "normalized": "a"},
            "experian": {"raw": None, "normalized": None},
        },
    }

    decision, rationale, info = _enforce_conditional_gate(
        "account_rating", "strong", "Single bureau", pack_line
    )

    assert decision == "no_case"
    assert "conditional_gate" in rationale
    assert info is not None
    assert info["reason"] == "insufficient_evidence"


def test_account_rating_gate_allows_conflict_with_evidence() -> None:
    pack_line = {
        "field": "account_rating",
        "conditional_gate": True,
        "min_corroboration": 2,
        "bureaus": {
            "transunion": {"raw": "A", "normalized": "a"},
            "experian": {"raw": "B", "normalized": "b"},
        },
    }

    decision, rationale, info = _enforce_conditional_gate(
        "account_rating", "strong", "Conflicting ratings", pack_line
    )

    assert decision == "strong"
    assert rationale == "Conflicting ratings"
    assert info is None
