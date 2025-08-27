from backend.core.logic.report_analysis.text_normalization import (
    NormalizationStats,
    normalize_page,
)


def test_whitespace_normalization() -> None:
    txt, _ = normalize_page("A B\tC\nD")
    assert txt == "A B C D"


def test_date_normalization() -> None:
    txt, stats = normalize_page("03/10/2019 2019/3/7")
    assert "2019-03-10" in txt and "2019-03-07" in txt
    assert stats.dates_converted == 2


def test_amount_normalization() -> None:
    txt, stats = normalize_page("$1,234.56 and \u20aa 7,890")
    assert "1234.56" in txt and "7890" in txt
    assert stats.amounts_converted == 2


def test_bidi_marker_stripped() -> None:
    txt, stats = normalize_page("\u202AABC\u202C")
    assert txt == "ABC"
    assert stats.bidi_stripped == 2


def test_idempotent() -> None:
    txt1, _ = normalize_page("$1,234.56 03/10/2019")
    txt2, _ = normalize_page(txt1)
    assert txt1 == txt2


def test_telemetry_counts() -> None:
    sample = "\u202A03/10/2019\u202C $1,234  A  B"
    _, stats = normalize_page(sample)
    assert stats.dates_converted == 1
    assert stats.amounts_converted == 1
    assert stats.bidi_stripped == 2
    assert stats.space_reduced_chars > 0

