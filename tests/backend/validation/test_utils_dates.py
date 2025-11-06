from datetime import date

import pytest

from backend.validation.utils_dates import business_to_calendar, business_to_calendar_days


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


def test_business_to_calendar_handles_friday_start():
    # Friday start should require a full weekend skip for the second day.
    assert business_to_calendar(4, 1) == 1
    assert business_to_calendar(4, 2) == 4


def test_business_to_calendar_spans_multiple_weekends():
    assert business_to_calendar(0, 12) == 16  # Two full weekends skipped.


def test_business_to_calendar_skips_holidays():
    july_3 = date(2023, 7, 3)  # Monday
    holidays = {date(2023, 7, 4)}  # Tuesday holiday
    assert business_to_calendar(july_3, 3, holidays=holidays) == 4


def test_business_to_calendar_days_rejects_invalid():
    with pytest.raises(ValueError):
        business_to_calendar_days("not-a-number")


def test_business_to_calendar_rejects_invalid_weekday():
    with pytest.raises(ValueError):
        business_to_calendar(7, 3)
