import pytest

from backend.validation.utils_dates import business_to_calendar_days


@pytest.mark.parametrize(
    ("business_days", "expected"),
    [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
        (5, 5),
        (6, 8),
        (10, 12),
        (14, 18),
        (19, 25),
    ],
)
def test_business_to_calendar_days_conversion(business_days, expected):
    assert business_to_calendar_days(business_days) == expected


def test_business_to_calendar_days_rejects_invalid():
    with pytest.raises(ValueError):
        business_to_calendar_days("not-a-number")
