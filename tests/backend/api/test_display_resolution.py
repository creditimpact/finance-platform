from __future__ import annotations

import sys
import types

if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")

    class _StubSession:
        def get(self, *_args: object, **_kwargs: object) -> None:  # pragma: no cover - stub
            raise RuntimeError("requests stub invoked")

        def close(self) -> None:  # pragma: no cover - stub
            pass

    class _StubRequestException(Exception):
        pass

    requests_stub.Session = _StubSession
    requests_stub.RequestException = _StubRequestException
    sys.modules["requests"] = requests_stub

from backend.api.app import (
    pick_account_number,
    pick_majority,
    resolve_display_fields,
)


def test_pick_majority_prefers_majority_with_precedence_source() -> None:
    value, source, method = pick_majority(
        {"equifax": "Open", "experian": "Closed", "transunion": "Closed"}
    )

    assert value == "Closed"
    assert source == "experian"
    assert method == "majority"


def test_pick_majority_uses_precedence_when_tied() -> None:
    value, source, method = pick_majority(
        {"equifax": "--", "experian": "Open", "transunion": "Closed"}
    )

    assert value == "Open"
    assert source == "experian"
    assert method == "precedence"


def test_pick_majority_reports_general_answer() -> None:
    value, source, method = pick_majority(
        {"equifax": "", "experian": "", "transunion": "Auto Loan"}
    )

    assert value == "Auto Loan"
    assert source == "transunion"
    assert method == "general"


def test_pick_account_number_prefers_most_digits_then_precedence() -> None:
    value, source = pick_account_number(
        {"equifax": "12-34", "experian": "A1 2 3 4", "transunion": "123456"}
    )

    assert value == "123456"
    assert source == "transunion"

    tie_value, tie_source = pick_account_number(
        {"equifax": "1111", "experian": "12-34-56", "transunion": "12 34 56"}
    )

    assert tie_value == "12-34-56"
    assert tie_source == "experian"


def test_resolve_display_fields_combines_rules() -> None:
    display = {
        "account_number": {
            "per_bureau": {
                "equifax": "12 34 56",
                "experian": "12-34-5678",
                "transunion": "123456789",
            }
        },
        "account_type": {
            "per_bureau": {
                "equifax": "Collection",
                "experian": "Auto Loan",
                "transunion": "Auto Loan",
            }
        },
        "status": {
            "per_bureau": {
                "equifax": "Open",
                "experian": "Closed",
                "transunion": "Closed",
            }
        },
        "balance_owed": {
            "per_bureau": {
                "equifax": "--",
                "experian": "$5,912",
                "transunion": "$5,912 ",
            }
        },
        "date_opened": {
            "per_bureau": {
                "equifax": "--",
                "experian": "09/01/2019",
                "transunion": "09/01/2019",
            }
        },
        "closed_date": {
            "equifax": "2020-02-12",
            "experian": "",
            "transunion": "02/12/2020",
        },
    }

    resolved = resolve_display_fields(display)

    assert resolved["account_number"] == {
        "value": "123456789",
        "source": "transunion",
        "method": "max_digits",
    }
    assert resolved["account_type"] == {
        "value": "Auto Loan",
        "source": "experian",
        "method": "majority",
    }
    assert resolved["status"] == {
        "value": "Closed",
        "source": "experian",
        "method": "majority",
    }
    assert resolved["balance_owed"] == {
        "value": "$5,912",
        "source": "experian",
        "method": "majority",
    }
    assert resolved["date_opened"] == {
        "value": "2019-09-01",
        "source": "experian",
        "method": "majority",
    }
    assert resolved["closed_date"] == {
        "value": "2020-02-12",
        "source": "transunion",
        "method": "precedence",
    }


def test_resolve_display_fields_handles_missing_values() -> None:
    resolved = resolve_display_fields({})

    assert resolved["account_type"] == {"value": "", "source": "", "method": "empty"}
    assert resolved["account_number"] == {
        "value": "",
        "source": "",
        "method": "max_digits",
    }
