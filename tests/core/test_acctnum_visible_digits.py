from backend.core.merge.acctnum import acctnum_visible_match


def test_visible_digits_prefix_match() -> None:
    ok, debug = acctnum_visible_match("349992*****", "3499921234567")
    assert ok
    assert debug["short"] == "349992"
    assert debug["long"] == "3499921234567"
    assert debug["match_offset"] == "0"
    assert debug["why"] == ""


def test_visible_digits_suffix_match() -> None:
    ok, debug = acctnum_visible_match("****6789", "123456789")
    assert ok
    assert debug["short"] == "6789"
    assert debug["long"] == "123456789"
    assert debug["match_offset"] == "5"
    assert debug["why"] == ""


def test_visible_digits_conflict() -> None:
    ok, debug = acctnum_visible_match("555550*****", "555555*****")
    assert not ok
    assert debug["short"] == "555550"
    assert debug["long"] == "555555"
    assert debug["match_offset"] == ""
    assert debug["why"] == "visible_digits_conflict"
