import os
import json
import logging
import importlib
from pathlib import Path

# Ensure triad parsing is enabled and tokens joined with spaces
os.environ["RAW_TRIAD_FROM_X"] = "1"
os.environ["RAW_JOIN_TOKENS_WITH_SPACE"] = "1"
import backend.config  # type: ignore
importlib.reload(backend.config)
from scripts import split_accounts_from_tsv  # type: ignore
importlib.reload(split_accounts_from_tsv)


def _run(rows: list[str], tmp_path: Path, caplog):
    header = "page\tline\ty0\ty1\tx0\tx1\ttext\n"
    tsv_path = tmp_path / "_debug_full.tsv"
    tsv_path.write_text(header + "".join(rows), encoding="utf-8")
    json_path = tmp_path / "accounts_from_full.json"
    with caplog.at_level(logging.INFO):
        split_accounts_from_tsv.main([
            "--full",
            str(tsv_path),
            "--json_out",
            str(json_path),
        ])
    data = json.loads(json_path.read_text())
    return data


def test_header_above(tmp_path: Path, caplog) -> None:
    rows = [
        # header exactly one line above anchor
        "1\t1\t10\t11\t50\t100\tTransUnion\n",
        "1\t1\t10\t11\t150\t200\tExperian\n",
        "1\t1\t10\t11\t250\t300\tEquifax\n",
        # anchor line
        "1\t2\t20\t21\t0\t10\tAccount #\n",
        "1\t2\t20\t21\t100\t110\tTransUnion\n",
        "1\t2\t20\t21\t200\t210\tExperian\n",
        "1\t2\t20\t21\t300\t310\tEquifax\n",
        # account number values
        "1\t3\t30\t31\t100\t110\tTU123\n",
        "1\t3\t30\t31\t200\t210\tXP123\n",
        "1\t3\t30\t31\t300\t310\tEQ123\n",
        # field row
        "1\t4\t40\t41\t0\t10\tAccount Status:\n",
        "1\t4\t40\t41\t100\t110\tOpen\n",
        "1\t4\t40\t41\t200\t210\tClosed\n",
        "1\t4\t40\t41\t300\t310\tCurrent\n",
    ]
    data = _run(rows, tmp_path, caplog)
    acc = data["accounts"][0]
    triad_fields = acc["triad_fields"]
    assert "TU123" in triad_fields["transunion"]["account_number_display"]
    assert triad_fields["experian"]["account_status"] == "Closed"
    assert "TRIAD_HEADER_XMIDS" in caplog.text


def test_cross_page(tmp_path: Path, caplog) -> None:
    rows = [
        # header on page 1
        "1\t1\t10\t11\t50\t100\tTransUnion\n",
        "1\t1\t10\t11\t150\t200\tExperian\n",
        "1\t1\t10\t11\t250\t300\tEquifax\n",
        # anchor on page 2
        "2\t1\t20\t21\t0\t10\tAccount #\n",
        "2\t1\t20\t21\t100\t110\tTransUnion\n",
        "2\t1\t20\t21\t200\t210\tExperian\n",
        "2\t1\t20\t21\t300\t310\tEquifax\n",
        # account numbers on page 2
        "2\t2\t30\t31\t100\t110\tTU999\n",
        "2\t2\t30\t31\t200\t210\tXP999\n",
        "2\t2\t30\t31\t300\t310\tEQ999\n",
        # field on page 2
        "2\t3\t40\t41\t0\t10\tAccount Status:\n",
        "2\t3\t40\t41\t100\t110\tOpen\n",
        "2\t3\t40\t41\t200\t210\tOpen\n",
        "2\t3\t40\t41\t300\t310\tOpen\n",
        # field on page 3
        "3\t1\t50\t51\t0\t10\tAccount Type:\n",
        "3\t1\t50\t51\t100\t110\tMortgage\n",
        "3\t1\t50\t51\t200\t210\tMortgage\n",
        "3\t1\t50\t51\t300\t310\tMortgage\n",
    ]
    data = _run(rows, tmp_path, caplog)
    acc = data["accounts"][0]
    triad_fields = acc["triad_fields"]
    assert triad_fields["transunion"]["account_type"] == "Mortgage"
    assert "TRIAD_CARRY_PAGE" in caplog.text


def test_two_year_no_colon(tmp_path: Path, caplog) -> None:
    rows = [
        # header
        "1\t1\t10\t11\t50\t100\tTransUnion\n",
        "1\t1\t10\t11\t150\t200\tExperian\n",
        "1\t1\t10\t11\t250\t300\tEquifax\n",
        # anchor
        "1\t2\t20\t21\t0\t10\tAccount #\n",
        "1\t2\t20\t21\t100\t110\tTransUnion\n",
        "1\t2\t20\t21\t200\t210\tExperian\n",
        "1\t2\t20\t21\t300\t310\tEquifax\n",
        # account numbers
        "1\t3\t30\t31\t100\t110\tTU123\n",
        "1\t3\t30\t31\t200\t210\tXP123\n",
        "1\t3\t30\t31\t300\t310\tEQ123\n",
        # payment status
        "1\t4\t40\t41\t0\t10\tPayment Status:\n",
        "1\t4\t40\t41\t100\t110\tOK\n",
        "1\t4\t40\t41\t200\t210\tOK\n",
        "1\t4\t40\t41\t300\t310\tOK\n",
        # sentinel without colon
        "1\t5\t50\t51\t0\t20\tTwo Year Payment History\n",
        # grid row that must be ignored
        "1\t6\t60\t61\t0\t10\tOK\n",
        "1\t6\t60\t61\t100\t110\t30\n",
        "1\t6\t60\t61\t200\t210\t60\n",
        "1\t6\t60\t61\t300\t310\t90\n",
    ]
    data = _run(rows, tmp_path, caplog)
    acc = data["accounts"][0]
    assert len(acc["triad_rows"]) == 2
    assert acc["triad_fields"]["transunion"]["payment_status"] == "OK"
    assert "TRIAD_STOP reason=two_year_history" in caplog.text


def test_bad_header_below(tmp_path: Path, caplog) -> None:
    rows = [
        # anchor without header above
        "1\t1\t20\t21\t0\t10\tAccount #\n",
        "1\t1\t20\t21\t100\t110\tTransUnion\n",
        "1\t1\t20\t21\t200\t210\tExperian\n",
        "1\t1\t20\t21\t300\t310\tEquifax\n",
        # header below anchor
        "1\t2\t30\t31\t50\t100\tTransUnion\n",
        "1\t2\t30\t31\t150\t200\tExperian\n",
        "1\t2\t30\t31\t250\t300\tEquifax\n",
        # field line
        "1\t3\t40\t41\t0\t10\tAccount Status:\n",
        "1\t3\t40\t41\t100\t110\tOpen\n",
        "1\t3\t40\t41\t200\t210\tOpen\n",
        "1\t3\t40\t41\t300\t310\tOpen\n",
    ]
    data = _run(rows, tmp_path, caplog)
    acc = data["accounts"][0]
    assert acc["triad_rows"] == []
    assert (
        "TRIAD_NO_HEADER_ABOVE_ANCHOR" in caplog.text
        or "TRIAD_STOP reason=layout_mismatch" in caplog.text
    )


def test_exact_labels(tmp_path: Path, caplog) -> None:
    rows = [
        # header
        "1\t1\t10\t11\t50\t100\tTransUnion\n",
        "1\t1\t10\t11\t150\t200\tExperian\n",
        "1\t1\t10\t11\t250\t300\tEquifax\n",
        # anchor
        "1\t2\t20\t21\t0\t10\tAccount #\n",
        "1\t2\t20\t21\t100\t110\tTransUnion\n",
        "1\t2\t20\t21\t200\t210\tExperian\n",
        "1\t2\t20\t21\t300\t310\tEquifax\n",
        # account numbers
        "1\t3\t30\t31\t100\t110\t1111\n",
        "1\t3\t30\t31\t200\t210\t2222\n",
        "1\t3\t30\t31\t300\t310\t3333\n",
        # account type
        "1\t4\t40\t41\t0\t10\tAccount Type:\n",
        "1\t4\t40\t41\t100\t110\tCredit Card\n",
        "1\t4\t40\t41\t200\t210\tLoan\n",
        "1\t4\t40\t41\t300\t310\tMortgage\n",
        # account status
        "1\t5\t50\t51\t0\t10\tAccount Status:\n",
        "1\t5\t50\t51\t100\t110\tOpen\n",
        "1\t5\t50\t51\t200\t210\tClosed\n",
        "1\t5\t50\t51\t300\t310\tPaid\n",
        # payment status
        "1\t6\t60\t61\t0\t10\tPayment Status:\n",
        "1\t6\t60\t61\t100\t110\tCurrent\n",
        "1\t6\t60\t61\t200\t210\tLate\n",
        "1\t6\t60\t61\t300\t310\tOK\n",
        # unknown label
        "1\t7\t70\t71\t0\t10\tWeird Label:\n",
        "1\t7\t70\t71\t100\t110\tX\n",
        "1\t7\t70\t71\t200\t210\tY\n",
        "1\t7\t70\t71\t300\t310\tZ\n",
    ]
    data = _run(rows, tmp_path, caplog)
    acc = data["accounts"][0]
    tf = acc["triad_fields"]["transunion"]
    assert "1111" in tf["account_number_display"]
    assert tf["account_type"] == "Credit Card"
    assert tf["account_status"] == "Open"
    assert tf["payment_status"] == "Current"
    assert len(acc["triad_rows"]) == 4
    assert "TRIAD_GUARD_SKIP" in caplog.text
