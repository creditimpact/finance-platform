import pytest

from backend.core.logic.decision_matrix import lookup_decision


@pytest.mark.parametrize(
    "field, reason_code, expected_decision",
    [
        ("account_status", "C4_ANY_CONFLICT", "strong_actionable"),
        ("payment_amount", "C1_TOO_LOW", "supportive_needs_companion"),
        ("account_type", "C2_MISMATCH", "neutral_context_only"),
        ("account_number_display", "C5_MASK_DIFF", "neutral_context_only"),
        ("two_year_payment_history", "C3_MISMATCH", "strong_actionable"),
    ],
)
def test_decision_matrix_basic(field, reason_code, expected_decision):
    """Ensure static matrix routes each (field, C-code) to the expected decision."""

    assert lookup_decision(field, reason_code) == expected_decision
