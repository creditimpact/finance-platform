import importlib
from pathlib import Path
from typing import Dict, List

import pytest

from backend.core.logic.report_analysis.triad_layout import TriadLayout
from backend.pipeline.runs import RUNS_ROOT_ENV

from tests.test_split_accounts_from_tsv import _run_split


TU_X0 = 172.875
XP_X0 = 309.0
EQ_X0 = 445.125
COLON_LEFT = TU_X0 - 0.3
COLON_RIGHT = TU_X0 + 0.3


def _token(line: int, x0: float, x1: float, text: str) -> dict:
    y0 = line * 10
    y1 = y0 + 1
    return {
        "page": 1,
        "line": line,
        "y0": y0,
        "y1": y1,
        "x0": x0,
        "x1": x1,
        "text": text,
    }


def _layout() -> TriadLayout:
    return TriadLayout(
        page=1,
        label_band=(0.0, TU_X0),
        tu_band=(TU_X0, XP_X0),
        xp_band=(XP_X0, EQ_X0),
        eq_band=(EQ_X0, float("inf")),
        label_right_x0=TU_X0,
        tu_left_x0=TU_X0,
        xp_left_x0=XP_X0,
        eq_left_x0=EQ_X0,
    )


def _write_tsv(path: Path, tokens: List[dict]) -> None:
    header = "page\tline\ty0\ty1\tx0\tx1\ttext\n"
    rows = [
        (
            f"{tok['page']}\t{tok['line']}\t{tok['y0']}\t{tok['y1']}\t"
            f"{tok['x0']:.3f}\t{tok['x1']:.3f}\t{tok['text']}\n"
        )
        for tok in tokens
    ]
    path.write_text(header + "".join(rows), encoding="utf-8")


def _account8_tokens() -> List[dict]:
    tokens: List[dict] = []
    # Triad header establishes the column seams directly from header x0 values.
    tokens.append(_token(1, TU_X0, TU_X0 + 30.0, "TransUnion"))
    tokens.append(_token(1, XP_X0, XP_X0 + 30.0, "Experian"))
    tokens.append(_token(1, EQ_X0, EQ_X0 + 30.0, "Equifax"))

    # Anchor row with a single Account # label token.
    tokens.append(_token(2, 100.0, 170.0, "Account #"))
    tokens.append(_token(2, TU_X0 + 10.0, TU_X0 + 50.0, "****"))
    tokens.append(_token(2, XP_X0 + 20.0, XP_X0 + 60.0, "XP-ACC"))
    tokens.append(_token(2, EQ_X0 + 20.0, EQ_X0 + 60.0, "EQ-ACC"))

    # High Balance
    tokens.append(_token(3, 90.0, 140.0, "High"))
    tokens.append(_token(3, 140.0, 170.0, "Balance"))
    tokens.append(_token(3, COLON_LEFT, COLON_RIGHT, ":"))
    tokens.append(_token(3, TU_X0 + 20.0, TU_X0 + 70.0, "$12,028"))
    tokens.append(_token(3, XP_X0 + 20.0, XP_X0 + 70.0, "$0"))
    tokens.append(_token(3, EQ_X0 + 20.0, EQ_X0 + 70.0, "$6,217"))

    # Last Verified
    tokens.append(_token(4, 90.0, 150.0, "Last"))
    tokens.append(_token(4, 150.0, 190.0, "Verified"))
    tokens.append(_token(4, COLON_LEFT, COLON_RIGHT, ":"))
    tokens.append(_token(4, TU_X0 + 20.0, TU_X0 + 70.0, "11.8.2025"))
    tokens.append(_token(4, XP_X0 + 20.0, XP_X0 + 70.0, "--"))
    tokens.append(_token(4, EQ_X0 + 20.0, EQ_X0 + 70.0, "--"))

    # Date of Last Activity
    tokens.append(_token(5, 60.0, 90.0, "Date"))
    tokens.append(_token(5, 90.0, 110.0, "of"))
    tokens.append(_token(5, 110.0, 150.0, "Last"))
    tokens.append(_token(5, 150.0, 200.0, "Activity"))
    tokens.append(_token(5, COLON_LEFT, COLON_RIGHT, ":"))
    tokens.append(_token(5, TU_X0 + 20.0, TU_X0 + 70.0, "30.3.2024"))
    tokens.append(_token(5, XP_X0 + 20.0, XP_X0 + 70.0, "1.6.2025"))
    tokens.append(_token(5, EQ_X0 + 20.0, EQ_X0 + 70.0, "1.2.2025"))

    # Date Reported
    tokens.append(_token(6, 60.0, 120.0, "Date"))
    tokens.append(_token(6, 120.0, 170.0, "Reported"))
    tokens.append(_token(6, COLON_LEFT, COLON_RIGHT, ":"))
    tokens.append(_token(6, TU_X0 + 20.0, TU_X0 + 70.0, "11.8.2025"))
    tokens.append(_token(6, XP_X0 + 20.0, XP_X0 + 70.0, "4.8.2025"))
    tokens.append(_token(6, EQ_X0 + 20.0, EQ_X0 + 70.0, "1.8.2025"))

    # Date Opened
    tokens.append(_token(7, 60.0, 120.0, "Date"))
    tokens.append(_token(7, 120.0, 170.0, "Opened"))
    tokens.append(_token(7, COLON_LEFT, COLON_RIGHT, ":"))
    tokens.append(_token(7, TU_X0 + 20.0, TU_X0 + 70.0, "20.11.2021"))
    tokens.append(_token(7, XP_X0 + 20.0, XP_X0 + 70.0, "1.11.2021"))
    tokens.append(_token(7, EQ_X0 + 20.0, EQ_X0 + 70.0, "1.11.2021"))

    # Closed Date
    tokens.append(_token(8, 80.0, 140.0, "Closed"))
    tokens.append(_token(8, 140.0, 180.0, "Date"))
    tokens.append(_token(8, COLON_LEFT, COLON_RIGHT, ":"))
    tokens.append(_token(8, TU_X0 + 20.0, TU_X0 + 70.0, "30.3.2024"))
    tokens.append(_token(8, XP_X0 + 20.0, XP_X0 + 70.0, "--"))
    tokens.append(_token(8, EQ_X0 + 20.0, EQ_X0 + 70.0, "30.3.2024"))

    return tokens


@pytest.fixture(autouse=True)
def _runs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runs_root = tmp_path / "runs"
    monkeypatch.setenv(RUNS_ROOT_ENV, str(runs_root))


@pytest.fixture
def split_module(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TRIAD_BAND_BY_X0", "1")
    monkeypatch.setenv("RAW_JOIN_TOKENS_WITH_SPACE", "1")
    monkeypatch.setenv("RAW_TRIAD_FROM_X", "1")
    monkeypatch.setenv("STAGEA_LABEL_PREFIX_MATCH", "1")
    monkeypatch.setenv("STAGEA_COLONLESS_TU_SPLIT", "1")
    monkeypatch.setenv("STAGEA_COLONLESS_TU_BOUNDARY", "1")
    monkeypatch.setenv("STAGEA_COLONLESS_TU_TEXT_FALLBACK", "1")
    mod = importlib.import_module("scripts.split_accounts_from_tsv")
    return importlib.reload(mod)


def test_tu_values_route_correctly_from_header_x0(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TRIAD_BAND_BY_X0", "1")
    tsv_path = tmp_path / "account8.tsv"
    _write_tsv(tsv_path, _account8_tokens())

    data, _accounts_dir, _sid = _run_split(tsv_path, caplog)
    triad_fields = data["accounts"][0]["triad_fields"]
    tu = triad_fields["transunion"]

    assert tu["account_number_display"] == "****"
    assert tu["high_balance"] == "$12,028"
    assert tu["last_verified"] == "11.8.2025"
    assert tu["date_of_last_activity"] == "30.3.2024"
    assert tu["date_reported"] == "11.8.2025"
    assert tu["date_opened"] == "20.11.2021"
    assert tu["closed_date"] == "30.3.2024"


def test_eq_is_capped_by_next_label_x0(split_module) -> None:
    layout = _layout()
    triad_fields: Dict[str, Dict[str, str]] = {
        "transunion": {},
        "experian": {},
        "equifax": {},
    }
    triad_order = ["transunion", "experian", "equifax"]

    first_row = [
        _token(3, 90.0, 130.0, "Closed"),
        _token(3, 130.0, 170.0, "Date"),
        _token(3, COLON_LEFT, COLON_RIGHT, ":"),
        _token(3, TU_X0 + 20.0, TU_X0 + 60.0, "30.3.2024"),
        _token(3, XP_X0 + 20.0, XP_X0 + 60.0, "--"),
        _token(3, EQ_X0 + 10.0, EQ_X0 + 60.0, "30.3.2024"),
        _token(3, EQ_X0 + 80.0, EQ_X0 + 140.0, "Last Payment:"),
    ]
    row1 = split_module.process_triad_labeled_line(
        first_row,
        layout,
        split_module.LABEL_MAP,
        None,
        triad_fields,
        triad_order,
    )
    assert row1["values"]["equifax"] == "30.3.2024"
    assert triad_fields["equifax"]["closed_date"] == "30.3.2024"

    second_row = [
        _token(4, 100.0, 170.0, "Last Payment:"),
        _token(4, TU_X0 + 20.0, TU_X0 + 60.0, "18.6.2025"),
        _token(4, XP_X0 + 20.0, XP_X0 + 60.0, "18.6.2025"),
        _token(4, EQ_X0 + 10.0, EQ_X0 + 60.0, "1.2.2025"),
    ]
    row2 = split_module.process_triad_labeled_line(
        second_row,
        layout,
        split_module.LABEL_MAP,
        None,
        triad_fields,
        triad_order,
    )
    assert row2["values"]["equifax"] == "1.2.2025"


def test_cleaning_empty_vs_dashes_and_masks(split_module) -> None:
    layout = _layout()
    triad_fields: Dict[str, Dict[str, str]] = {
        "transunion": {},
        "experian": {},
        "equifax": {},
    }
    triad_order = ["transunion", "experian", "equifax"]

    # Masked account numbers must be preserved for every bureau.
    account_row = [
        _token(2, 100.0, 170.0, "Account #"),
        _token(2, TU_X0 + 20.0, TU_X0 + 60.0, "****"),
        _token(2, XP_X0 + 20.0, XP_X0 + 60.0, "****"),
        _token(2, EQ_X0 + 20.0, EQ_X0 + 80.0, "552433**********"),
    ]
    split_module.process_triad_labeled_line(
        account_row,
        layout,
        split_module.LABEL_MAP,
        None,
        triad_fields,
        triad_order,
    )

    # Dashes must remain dashes when present in the source.
    dash_row = [
        _token(3, 90.0, 140.0, "High"),
        _token(3, 140.0, 170.0, "Balance"),
        _token(3, COLON_LEFT, COLON_RIGHT, ":"),
        _token(3, TU_X0 + 20.0, TU_X0 + 70.0, "$12,028"),
        _token(3, XP_X0 + 20.0, XP_X0 + 60.0, "--"),
        _token(3, EQ_X0 + 20.0, EQ_X0 + 60.0, "--"),
    ]
    split_module.process_triad_labeled_line(
        dash_row,
        layout,
        split_module.LABEL_MAP,
        None,
        triad_fields,
        triad_order,
    )

    # Missing tokens should stay as empty strings rather than being coerced to dashes.
    empty_row = [
        _token(4, 90.0, 150.0, "Credit"),
        _token(4, 150.0, 200.0, "Limit"),
        _token(4, COLON_LEFT, COLON_RIGHT, ":"),
        _token(4, TU_X0 + 20.0, TU_X0 + 60.0, "$5,000"),
        _token(4, EQ_X0 + 20.0, EQ_X0 + 60.0, "$7,500"),
    ]
    split_module.process_triad_labeled_line(
        empty_row,
        layout,
        split_module.LABEL_MAP,
        None,
        triad_fields,
        triad_order,
    )

    assert triad_fields["transunion"]["account_number_display"] == "****"
    assert triad_fields["experian"]["account_number_display"] == "****"
    assert triad_fields["equifax"]["account_number_display"] == "552433**********"
    assert triad_fields["experian"]["high_balance"] == "--"
    assert triad_fields["equifax"]["high_balance"] == "--"
    assert triad_fields["experian"]["credit_limit"] == ""


def test_original_creditor_prefix_rescue_preserves_tu_value(split_module) -> None:
    layout = _layout()
    triad_fields: Dict[str, Dict[str, str]] = {
        "transunion": {},
        "experian": {},
        "equifax": {},
    }
    triad_order = ["transunion", "experian", "equifax"]

    colonless_row = [
        _token(12, 80.0, 130.0, "Original"),
        _token(12, 130.0, 190.0, "Creditor"),
        _token(12, 165.0, 170.0, "01"),
        _token(12, TU_X0 + 5.0, TU_X0 + 70.0, "PALISADES"),
        _token(12, TU_X0 + 70.0, TU_X0 + 120.0, "FUNDING"),
        _token(12, TU_X0 + 120.0, TU_X0 + 150.0, "CORP"),
        _token(12, XP_X0 + 10.0, XP_X0 + 20.0, "--"),
        _token(12, EQ_X0 + 10.0, EQ_X0 + 20.0, "--"),
    ]

    split_module.process_triad_labeled_line(
        colonless_row,
        layout,
        split_module.LABEL_MAP,
        None,
        triad_fields,
        triad_order,
    )

    assert triad_fields["transunion"]["original_creditor"] == "PALISADES FUNDING CORP"
    assert triad_fields["experian"]["original_creditor"] == "--"
    assert triad_fields["equifax"]["original_creditor"] == "--"


def test_original_creditor_colonless_text_fallback(split_module) -> None:
    layout = _layout()
    triad_fields: Dict[str, Dict[str, str]] = {
        "transunion": {},
        "experian": {},
        "equifax": {},
    }
    triad_order = ["transunion", "experian", "equifax"]

    text_only_row = [
        _token(13, 70.0, 120.0, "Original"),
        _token(13, 120.0, 170.0, "Creditor"),
        _token(13, 160.0, 165.0, "01"),
        _token(13, 166.0, 168.0, "PALISADES"),
        _token(13, 168.0, 170.0, "FUNDING"),
        _token(13, 170.0, 171.5, "CORP"),
        _token(13, 171.5, 171.7, "--"),
        _token(13, 171.7, 171.9, "--"),
    ]

    split_module.process_triad_labeled_line(
        text_only_row,
        layout,
        split_module.LABEL_MAP,
        None,
        triad_fields,
        triad_order,
    )

    assert triad_fields["transunion"]["original_creditor"] == "PALISADES FUNDING CORP"
    assert triad_fields["experian"]["original_creditor"] == "--"
    assert triad_fields["equifax"]["original_creditor"] == "--"


def test_triad_label_variant_orig_abbrev(split_module) -> None:
    layout = _layout()
    triad_fields: Dict[str, Dict[str, str]] = {
        "transunion": {},
        "experian": {},
        "equifax": {},
    }
    triad_order = ["transunion", "experian", "equifax"]

    tokens = [
        _token(5, 100.0, 140.0, "Orig."),
        _token(5, 140.0, 190.0, "Creditor"),
        _token(5, COLON_LEFT, COLON_RIGHT, ":"),
        _token(5, TU_X0 + 15.0, TU_X0 + 70.0, "PALISADES FUNDING CORP"),
        _token(5, XP_X0 + 15.0, XP_X0 + 70.0, "ATLANTIC CAPITAL"),
        _token(5, EQ_X0 + 15.0, EQ_X0 + 70.0, "PACIFIC HOLDINGS"),
    ]

    split_module.process_triad_labeled_line(
        tokens,
        layout,
        split_module.LABEL_MAP,
        None,
        triad_fields,
        triad_order,
    )

    assert triad_fields["transunion"]["original_creditor"] == "PALISADES FUNDING CORP"
    assert triad_fields["experian"]["original_creditor"] == "ATLANTIC CAPITAL"
    assert triad_fields["equifax"]["original_creditor"] == "PACIFIC HOLDINGS"


def test_no_bleed_with_explicit_dashes(split_module) -> None:
    layout = _layout()
    triad_fields: Dict[str, Dict[str, str]] = {
        "transunion": {},
        "experian": {},
        "equifax": {},
    }
    triad_order = ["transunion", "experian", "equifax"]

    row = [
        _token(5, 80.0, 140.0, "Account"),
        _token(5, 140.0, 190.0, "Type"),
        _token(5, COLON_LEFT, COLON_RIGHT, ":"),
        _token(5, TU_X0 + 10.0, TU_X0 + 40.0, "Flexible"),
        _token(5, TU_X0 + 45.0, TU_X0 + 80.0, "spending"),
        _token(5, TU_X0 + 85.0, TU_X0 + 120.0, "credit"),
        _token(5, TU_X0 + 125.0, TU_X0 + 160.0, "card"),
        _token(5, XP_X0 + 25.0, XP_X0 + 55.0, "--"),
        _token(5, EQ_X0 + 20.0, EQ_X0 + 55.0, "--"),
    ]

    row_state = split_module.process_triad_labeled_line(
        row,
        layout,
        split_module.LABEL_MAP,
        None,
        triad_fields,
        triad_order,
    )

    assert row_state["values"]["transunion"] == "Flexible spending credit card"
    assert row_state["values"]["experian"] == "--"
    assert row_state["values"]["equifax"] == "--"

    assert triad_fields["transunion"]["account_type"] == "Flexible spending credit card"
    assert triad_fields["experian"]["account_type"] == "--"
    assert triad_fields["equifax"]["account_type"] == "--"

    tu_tokens = [t for t in row[3:7] if split_module._token_band(t, layout) == "tu"]
    assert tu_tokens, "TU tokens should stay within the TU band"
    xp_token = row[7]
    eq_token = row[8]
    assert split_module._token_band(xp_token, layout) == "xp"
    assert split_module._token_band(eq_token, layout) == "eq"


def test_numeric_rows_respect_guard(split_module) -> None:
    layout = _layout()
    triad_fields: Dict[str, Dict[str, str]] = {
        "transunion": {},
        "experian": {},
        "equifax": {},
    }
    triad_order = ["transunion", "experian", "equifax"]

    row = [
        _token(6, 90.0, 150.0, "High"),
        _token(6, 150.0, 200.0, "Balance"),
        _token(6, COLON_LEFT, COLON_RIGHT, ":"),
        _token(6, XP_X0 - 1.0, XP_X0 + 30.0, "1000"),
        _token(6, XP_X0 + 6.0, XP_X0 + 40.0, "2000"),
        _token(6, EQ_X0 + 10.0, EQ_X0 + 50.0, "3000"),
    ]

    row_state = split_module.process_triad_labeled_line(
        row,
        layout,
        split_module.LABEL_MAP,
        None,
        triad_fields,
        triad_order,
    )

    assert row_state["values"]["transunion"] == "1000"
    assert row_state["values"]["experian"] == "2000"
    assert row_state["values"]["equifax"] == "3000"

    assert triad_fields["transunion"]["high_balance"] == "1000"
    assert triad_fields["experian"]["high_balance"] == "2000"
    assert triad_fields["equifax"]["high_balance"] == "3000"


def test_tail_split_single_token_only(split_module) -> None:
    layout = _layout()
    triad_fields: Dict[str, Dict[str, str]] = {
        "transunion": {},
        "experian": {},
        "equifax": {},
    }
    triad_order = ["transunion", "experian", "equifax"]

    row = [
        _token(7, 90.0, 150.0, "High"),
        _token(7, 150.0, 200.0, "Balance"),
        _token(7, COLON_LEFT, COLON_RIGHT, ":"),
        _token(7, TU_X0 - 5.0, TU_X0 + 10.0, "400 500 600"),
    ]

    row_state = split_module.process_triad_labeled_line(
        row,
        layout,
        split_module.LABEL_MAP,
        None,
        triad_fields,
        triad_order,
    )

    assert row_state["values"]["transunion"] == "400"
    assert row_state["values"]["experian"] == "500"
    assert row_state["values"]["equifax"] == "600"

    assert triad_fields["transunion"]["high_balance"] == "400"
    assert triad_fields["experian"]["high_balance"] == "500"
    assert triad_fields["equifax"]["high_balance"] == "600"


def test_presence_semantics_for_empty_and_dashes(split_module) -> None:
    layout = _layout()
    triad_fields: Dict[str, Dict[str, str]] = {
        "transunion": {},
        "experian": {},
        "equifax": {},
    }
    triad_order = ["transunion", "experian", "equifax"]

    empty_row = [
        _token(8, 80.0, 140.0, "Payment"),
        _token(8, 140.0, 200.0, "Amount"),
        _token(8, COLON_LEFT, COLON_RIGHT, ":"),
        _token(8, TU_X0 + 20.0, TU_X0 + 55.0, "$75"),
    ]

    empty_state = split_module.process_triad_labeled_line(
        empty_row,
        layout,
        split_module.LABEL_MAP,
        None,
        triad_fields,
        triad_order,
    )

    dash_row = [
        _token(9, 80.0, 140.0, "Payment"),
        _token(9, 140.0, 200.0, "Frequency"),
        _token(9, COLON_LEFT, COLON_RIGHT, ":"),
        _token(9, TU_X0 + 20.0, TU_X0 + 55.0, "Weekly"),
        _token(9, XP_X0 + 25.0, XP_X0 + 55.0, "--"),
    ]

    dash_state = split_module.process_triad_labeled_line(
        dash_row,
        layout,
        split_module.LABEL_MAP,
        None,
        triad_fields,
        triad_order,
    )

    assert empty_state["values"]["experian"] == ""
    assert empty_state["values"]["equifax"] == ""
    assert dash_state["values"]["experian"] == "--"
    assert triad_fields["experian"]["payment_frequency"] == "--"


def test_account_type_tracing(split_module, monkeypatch: pytest.MonkeyPatch, caplog) -> None:
    # Enable triad debug logs so _triad_band_log emits messages
    monkeypatch.setenv("STAGEA_DEBUG", "1")
    monkeypatch.setenv("TRIAD_BAND_BY_X0", "1")

    layout = _layout()
    triad_fields: Dict[str, Dict[str, str]] = {
        "transunion": {},
        "experian": {},
        "equifax": {},
    }
    triad_order = ["transunion", "experian", "equifax"]

    row = [
        _token(5, 80.0, 140.0, "Account"),
        _token(5, 140.0, 190.0, "Type"),
        _token(5, COLON_LEFT, COLON_RIGHT, ":"),
        _token(5, TU_X0 + 10.0, TU_X0 + 40.0, "Flexible"),
        _token(5, TU_X0 + 45.0, TU_X0 + 80.0, "spending"),
        _token(5, TU_X0 + 85.0, TU_X0 + 120.0, "credit"),
        _token(5, TU_X0 + 125.0, TU_X0 + 160.0, "card"),
        _token(5, XP_X0 + 25.0, XP_X0 + 55.0, "--"),
        _token(5, EQ_X0 + 20.0, EQ_X0 + 55.0, "--"),
    ]

    with caplog.at_level("INFO"):
        row_state = split_module.process_triad_labeled_line(
            row,
            layout,
            split_module.LABEL_MAP,
            None,
            triad_fields,
            triad_order,
        )

    # Final assembled values
    assert row_state["values"]["transunion"] == "Flexible spending credit card"
    assert row_state["values"]["experian"] == "--"
    assert row_state["values"]["equifax"] == "--"

    # Trace should include header x0s via ROW_BANDS and per-token band assignments
    text = caplog.text
    assert "ROW_BANDS key=account_type" in text
    # 4 TU tokens assigned in band
    assert "TOK p=1 l=5" in text  # at least some token logs for this line
    assert text.count("-> TU text='Flexible'") == 1
    assert text.count("-> TU text='spending'") == 1
    assert text.count("-> TU text='credit'") == 1
    assert text.count("-> TU text='card'") == 1
    # XP/EQ are dashes
    assert "-> XP text='--'" in text
    assert "-> EQ text='--'" in text
