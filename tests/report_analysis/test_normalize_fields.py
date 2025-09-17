import pytest

from backend.core.logic.report_analysis.normalize_fields import clean_value


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("****", "****"),
        ("  ****  ", "****"),
        ("552433**********", "552433**********"),
        ("552433    **********", "552433 **********"),
    ],
)
def test_clean_value_preserves_masked_sequences(raw: str, expected: str) -> None:
    assert clean_value(raw) == expected


def test_clean_value_still_normalizes_non_masked_values() -> None:
    assert clean_value("  $1,200  ") == "$1,200"
    assert clean_value("--") == "--"
