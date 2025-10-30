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


def test_try_scan_raw_triads_for_original_creditor_returns_value() -> None:
    raw_lines = [
        {"text": "Account #: 123"},
        {"text": "Original Creditor 01 PALISADES FUNDING CORP -- --"},
    ]

    result = pcb.try_scan_raw_triads_for_original_creditor(raw_lines)

    assert result == "PALISADES FUNDING CORP"


def test_try_scan_raw_triads_for_original_creditor_ignores_missing_value() -> None:
    raw_lines = [
        {"text": "Original Creditor 01 -- -- --"},
        {"text": "Other Row"},
    ]

    result = pcb.try_scan_raw_triads_for_original_creditor(raw_lines)

    assert result is None


def test_post_extract_backfill_updates_transunion_original_creditor(monkeypatch) -> None:
    monkeypatch.setattr(pcb, "STAGEA_ORIGCRED_POST_EXTRACT", True)
    triad_fields = {
        "transunion": {"original_creditor": ""},
        "experian": {},
        "equifax": {},
    }
    raw_lines = [
        {"text": "Original Creditor 01 PALISADES FUNDING CORP -- --"},
    ]

    pcb._maybe_backfill_original_creditor(triad_fields, raw_lines)

    assert triad_fields["transunion"]["original_creditor"] == "PALISADES FUNDING CORP"


def test_post_extract_backfill_respects_existing_value(monkeypatch) -> None:
    monkeypatch.setattr(pcb, "STAGEA_ORIGCRED_POST_EXTRACT", True)
    triad_fields = {
        "transunion": {"original_creditor": "ALREADY THERE"},
    }
    raw_lines = [
        {"text": "Original Creditor 01 PALISADES FUNDING CORP -- --"},
    ]

    pcb._maybe_backfill_original_creditor(triad_fields, raw_lines)

    assert triad_fields["transunion"]["original_creditor"] == "ALREADY THERE"
