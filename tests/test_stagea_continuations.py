import importlib
from pathlib import Path
from typing import List

import pytest

from tests.test_split_accounts_from_tsv import _run_split
from backend.pipeline.runs import RUNS_ROOT_ENV


TU_X0 = 172.875
XP_X0 = 309.0
EQ_X0 = 445.125


def _w(tsv_path: Path, rows: List[str]) -> None:
    header = "page\tline\ty0\ty1\tx0\tx1\ttext\n"
    tsv_path.write_text(header + "".join(rows), encoding="utf-8")


def _header_and_anchor_rows() -> List[str]:
    # Header with rough mids to establish bands
    rows = [
        "1\t1\t10\t11\t150\t200\tTransUnion\n",
        "1\t1\t10\t11\t290\t330\tExperian\n",
        "1\t1\t10\t11\t430\t480\tEquifax\n",
        # Anchor line: left-x0 cutoffs for TU/XP/EQ columns
        "1\t2\t20\t21\t0\t20\tAccount #\n",
        f"1\t2\t20\t21\t{TU_X0}\t{TU_X0+20}\tTUANCH\n",
        f"1\t2\t20\t21\t{XP_X0}\t{XP_X0+20}\tXPANCH\n",
        f"1\t2\t20\t21\t{EQ_X0}\t{EQ_X0+20}\tEQANCH\n",
    ]
    return rows


@pytest.fixture(autouse=True)
def _triad_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TRIAD_BAND_BY_X0", "1")
    monkeypatch.setenv("RAW_TRIAD_FROM_X", "1")
    monkeypatch.setenv("RAW_JOIN_TOKENS_WITH_SPACE", "1")
    # Default guard as per task
    monkeypatch.setenv("TRIAD_BOUNDARY_GUARD", "2.0")


@pytest.fixture(autouse=True)
def _runs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    runs_root = tmp_path / "runs"
    monkeypatch.setenv(RUNS_ROOT_ENV, str(runs_root))


def test_continuation_eq_capped_no_bleed(tmp_path: Path, caplog: pytest.LogCaptureFixture):
    tsv = tmp_path / "_cont_eq.tsv"
    rows = _header_and_anchor_rows()
    # Labeled row: Creditor Remarks
    rows += [
        "1\t3\t30\t31\t0\t60\tCreditor Remarks:\n",
        f"1\t3\t30\t31\t{TU_X0+10}\t{TU_X0+35}\tCharged\n",
        f"1\t3\t30\t31\t{TU_X0+45}\t{TU_X0+70}\toff\n",
        f"1\t3\t30\t31\t{TU_X0+80}\t{TU_X0+100}\tas\n",
        f"1\t3\t30\t31\t{TU_X0+105}\t{TU_X0+125}\tbad\n",
        f"1\t3\t30\t31\t{TU_X0+130}\t{TU_X0+155}\tdebt\n",
        f"1\t3\t30\t31\t{XP_X0+20}\t{XP_X0+50}\t--\n",
        f"1\t3\t30\t31\t{EQ_X0+15}\t{EQ_X0+55}\taccount\n",
    ]
    # Continuation line: only EQ tokens, must not bleed into XP
    rows += [
        f"1\t4\t40\t41\t{EQ_X0+10}\t{EQ_X0+60}\tAccounts\n",
        f"1\t4\t40\t41\t{EQ_X0+65}\t{EQ_X0+100}\tclosed\n",
        f"1\t4\t40\t41\t{EQ_X0+90}\t{EQ_X0+110}\tby\n",
        f"1\t4\t40\t41\t{EQ_X0+105}\t{EQ_X0+120}\tcredit\n",
        f"1\t4\t40\t41\t{EQ_X0+115}\t{EQ_X0+118}\tgrantor\n",
    ]
    _w(tsv, rows)

    data, _acc_dir, _sid = _run_split(tsv, caplog)
    fields = data["accounts"][0]["triad_fields"]

    assert fields["transunion"]["creditor_remarks"] == "Charged off as bad debt"
    assert fields["experian"]["creditor_remarks"] == "--"
    assert fields["equifax"]["creditor_remarks"].startswith(
        "account Accounts closed"
    )


def test_tu_account_type_wrap_keeps_all_words(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    tsv = tmp_path / "_cont_tu.tsv"
    rows = _header_and_anchor_rows()
    # Labeled row: Account Type with first two words in TU; XP/EQ dashes
    rows += [
        "1\t3\t30\t31\t0\t40\tAccount\n",
        "1\t3\t30\t31\t40\t80\tType:\n",
        f"1\t3\t30\t31\t{TU_X0+10}\t{TU_X0+40}\tFlexible\n",
        f"1\t3\t30\t31\t{TU_X0+45}\t{TU_X0+85}\tspending\n",
        f"1\t3\t30\t31\t{XP_X0+20}\t{XP_X0+50}\t--\n",
        f"1\t3\t30\t31\t{EQ_X0+20}\t{EQ_X0+55}\t--\n",
    ]
    # Continuation line: TU gets remaining words "credit card"
    rows += [
        f"1\t4\t40\t41\t{TU_X0+60}\t{TU_X0+95}\tcredit\n",
        f"1\t4\t40\t41\t{TU_X0+85}\t{TU_X0+105}\tcard\n",
    ]
    _w(tsv, rows)

    data, _acc_dir, _sid = _run_split(tsv, caplog)
    fields = data["accounts"][0]["triad_fields"]

    assert fields["transunion"]["account_type"] == "Flexible spending credit card"
    assert fields["experian"]["account_type"] == "--"
    assert fields["equifax"]["account_type"] == "--"
