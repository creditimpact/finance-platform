import pytest

from backend.core.logic.report_analysis import problem_case_builder as pcb


def test_sanitize_preserves_populated_values() -> None:
    acc = {
        "triad_fields": {
            "transunion": {"triad_rows": [1], "original_creditor": "PALISADES"},
            "experian": "--",
            "equifax": None,
        }
    }

    result = pcb._build_bureaus_payload_from_stagea(acc)

    assert result["transunion"] == {"original_creditor": "PALISADES"}
    assert result["experian"] == "--"
    assert result["equifax"] == {}


@pytest.mark.parametrize(
    "payload,expected",
    [
        ({"triad_rows": []}, {}),
        ("value", "value"),
        (None, {}),
    ],
)
def test_sanitize_bureau_fields_respects_non_empty_values(payload, expected) -> None:
    assert pcb._sanitize_bureau_fields(payload) == expected
