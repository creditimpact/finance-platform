from backend.core.merge.acctnum import acctnum_match_visible, acctnum_level


def test_substring_left() -> None:
    ok, _ = acctnum_match_visible("349992*****", "3499921234567")
    assert ok
    level, debug = acctnum_level("349992*****", "3499921234567")
    assert level == "none"
    assert debug["why"] == "digit_conflict"


def test_substring_right() -> None:
    ok, _ = acctnum_match_visible("****6789", "123456789")
    assert ok
    level, debug = acctnum_level("****6789", "123456789")
    assert level == "none"
    assert debug["why"] == "digit_conflict"


def test_conflict_visible_digit() -> None:
    ok, _ = acctnum_match_visible("555550*****", "555555*****")
    assert not ok and acctnum_level("555550*****", "555555*****")[0] == "none"


def test_identical_full_numbers() -> None:
    ok, _ = acctnum_match_visible("349992123456789", "349992123456789")
    assert ok
