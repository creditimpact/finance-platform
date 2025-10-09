from backend.core.logic.decision_matrix import get_decision_matrix, lookup_decision


EXPECTED_FIELDS = {
    "account_number_display",
    "account_rating",
    "account_status",
    "account_type",
    "balance_owed",
    "closed_date",
    "credit_limit",
    "creditor_type",
    "date_of_last_activity",
    "date_opened",
    "date_reported",
    "high_balance",
    "last_payment",
    "last_verified",
    "past_due_amount",
    "payment_amount",
    "payment_frequency",
    "payment_status",
    "seven_year_history",
    "term_length",
    "two_year_payment_history",
}


def test_matrix_contains_expected_fields():
    matrix = get_decision_matrix()

    assert set(matrix) == EXPECTED_FIELDS
    for decisions in matrix.values():
        assert set(decisions) == {"C1", "C2", "C3", "C4", "C5"}


def test_lookup_decision_matches_matrix():
    assert (
        lookup_decision("account_status", "C3_TWO_PRESENT_CONFLICT")
        == "strong_actionable"
    )
    assert (
        lookup_decision("account_rating", "C2_ONE_MISSING")
        == "neutral_context_only"
    )
    assert lookup_decision("account_type", "C5_ALL_DIFF") == "supportive_needs_companion"


def test_lookup_decision_handles_missing_values():
    assert lookup_decision("balance_owed", "C6_ALL_MISSING") is None
    assert lookup_decision("unknown_field", "C1_TWO_PRESENT_ONE_MISSING") is None
    assert lookup_decision("payment_status", None) is None
