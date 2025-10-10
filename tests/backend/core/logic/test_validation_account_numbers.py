"""Tests for deterministic account-number validation behavior."""

import pytest

from backend.core.logic.validation_requirements import (
    _account_number_pair_metrics,
    build_validation_requirements,
)


@pytest.mark.parametrize(
    ("left", "right", "expected_relation"),
    [
        ("123456781234", "****1234", "match"),
        ("****1234", "XX1234", "match"),
        ("111111112345", "222222222345", "conflict"),
        ("ABC", "XYZ", "conflict"),
        ("ABC1234", "ending in 1234", "match"),
    ],
)
def test_account_number_pair_metrics_relations(left: str, right: str, expected_relation: str) -> None:
    level, relation = _account_number_pair_metrics(left, right)

    assert relation == expected_relation
    if expected_relation == "match":
        assert level == "exact_or_known_match"
    else:
        assert level == "none"


def test_account_number_validation_considers_partial_matches_consistent() -> None:
    bureaus = {
        "equifax": {"account_number_display": "123456781234"},
        "experian": {"account_number_display": "****1234"},
        "transunion": {"account_number_display": "XX1234"},
    }

    requirements, inconsistencies, _ = build_validation_requirements(bureaus)

    assert all(entry["field"] != "account_number_display" for entry in requirements)
    assert "account_number_display" not in inconsistencies


def test_account_number_validation_flags_conflicting_full_numbers() -> None:
    bureaus = {
        "equifax": {"account_number_display": "111111112345"},
        "experian": {"account_number_display": "222222222345"},
        "transunion": {"account_number_display": "****2345"},
    }

    requirements, inconsistencies, _ = build_validation_requirements(bureaus)

    conflict_fields = {entry["field"] for entry in requirements}
    assert "account_number_display" in conflict_fields

    account_details = inconsistencies["account_number_display"]
    assert "equifax" in account_details.get("disagreeing_bureaus", [])
    assert account_details["normalized"]["equifax"]["display"] == "111111112345"
    assert account_details["normalized"]["experian"]["display"] == "222222222345"


def test_account_number_validation_treats_partials_without_conflict_as_consistent() -> None:
    bureaus = {
        "equifax": {"account_number_display": "****1234"},
        "experian": {"account_number_display": "XX1234"},
        "transunion": {"account_number_display": "ending in 1234"},
    }

    requirements, inconsistencies, _ = build_validation_requirements(bureaus)

    assert all(entry["field"] != "account_number_display" for entry in requirements)
    assert "account_number_display" not in inconsistencies
