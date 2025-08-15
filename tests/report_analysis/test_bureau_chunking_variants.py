from backend.core.logic.report_analysis.report_prompting import _split_text_by_bureau
from backend.core.logic.utils.names_normalization import BUREAUS


def _compute(text: str):
    segments = _split_text_by_bureau(text)
    missing = [b for b in BUREAUS if b not in segments]
    return segments, missing


def test_odd_spacing_headings():
    text = (
        "EX PERIAN REPORT\n" "data\n" "\fEQUI FAX REPORT\n" "more\n" "\fTRANS UNION REPORT\n" "end"
    )
    segments, missing = _compute(text)
    assert set(segments) == set(BUREAUS)
    assert missing == []


def test_lowercase_headings():
    text = (
        "experian report\n" "a\n" "\fequifax report\n" "b\n" "\ftransunion report\n" "c"
    )
    segments, missing = _compute(text)
    assert set(segments) == set(BUREAUS)
    assert missing == []


def test_merged_sections_missing_bureau():
    text = (
        "Experian report\n" "foo\n" "Equifax report\n" "bar\n"
    )
    segments, missing = _compute(text)
    assert set(segments) == {"Experian", "Equifax"}
    assert missing == ["TransUnion"]
