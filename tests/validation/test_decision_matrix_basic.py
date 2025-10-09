"""Basic decision matrix expectations for validation findings."""

from __future__ import annotations

import pytest

from backend.validation.decision_matrix import decide_default


@pytest.mark.parametrize(
    "field, reason_code, expected",
    [
        ("account_status", "C4_TWO_MATCH_ONE_DIFF", "strong_actionable"),
        ("payment_amount", "C1_TWO_PRESENT_ONE_MISSING", "supportive_needs_companion"),
        ("account_type", "C2_ONE_MISSING", "neutral_context_only"),
        ("account_number_display", "C5_ALL_DIFF", "neutral_context_only"),
        ("two_year_payment_history", "C3_TWO_PRESENT_CONFLICT", "strong_actionable"),
    ],
)
def test_decide_default_expected_categories(field: str, reason_code: str, expected: str) -> None:
    """The deterministic matrix should map fields to the configured decision."""

    assert decide_default(field, reason_code) == expected
