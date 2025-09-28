from backend.core.merge.acctnum import acctnum_visible_match


def test_visible_digits_prefix_match() -> None:
    ok, debug = acctnum_visible_match("349992*****", "3499921234")
    assert ok
    assert debug["short"] == "349992"
    assert debug["long"].startswith("349992")


def test_visible_digits_suffix_match() -> None:
    ok, debug = acctnum_visible_match("****6789", "123456789")
    assert ok
    assert debug["short"] == "6789"
    assert debug["long"].endswith("6789")


def test_visible_digits_conflict() -> None:
    ok, debug = acctnum_visible_match("555550*****", "555555*****")
    assert not ok
    assert debug["why"] == "visible_digits_conflict"
