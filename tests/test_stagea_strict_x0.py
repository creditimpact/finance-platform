import json
import os
import uuid
from pathlib import Path

import pytest

from backend.pipeline.runs import RUNS_ROOT_ENV
from scripts import split_accounts_from_tsv


@pytest.fixture(autouse=True)
def _strict_env(tmp_path, monkeypatch):
    # Runs root
    runs_root = tmp_path / "runs"
    monkeypatch.setenv(RUNS_ROOT_ENV, str(runs_root))
    # Strict X0 behavior
    monkeypatch.setenv("RAW_TRIAD_FROM_X", "1")
    monkeypatch.setenv("TRIAD_BAND_BY_X0", "1")
    monkeypatch.setenv("TRIAD_X0_STRICT", "1")
    monkeypatch.setenv("TRIAD_CONT_USE_NEAREST", "0")
    monkeypatch.setenv("TRIAD_BOUNDARY_GUARD", "0")
    monkeypatch.setenv("TRIAD_X0_TOL", "0.5")


def _run_split(tsv_path: Path, sid: str | None = None):
    args = ["--full", str(tsv_path)]
    use_sid = sid or f"sid-{uuid.uuid4().hex}"
    args.extend(["--sid", use_sid])
    split_accounts_from_tsv.main(args)
    runs_root = Path(os.environ[RUNS_ROOT_ENV])
    accounts_dir = runs_root / use_sid / "traces" / "accounts_table"
    accounts_json = accounts_dir / "accounts_from_full.json"
    data = json.loads(accounts_json.read_text())
    return data, accounts_dir, use_sid


def _emit(p: Path, rows: list[tuple[int, int, float, float, float, float, str]]):
    # rows: (page, line, y0, y1, x0, x1, text)
    with p.open("w", encoding="utf-8") as fh:
        fh.write("page\tline\ty0\ty1\tx0\tx1\ttext\n")
        for pg, ln, y0, y1, x0, x1, txt in rows:
            fh.write(f"{pg}\t{ln}\t{y0}\t{y1}\t{x0}\t{x1}\t{txt}\n")


def _header_and_anchor_rows():
    # Header tokens (line 1), then anchor line (line 2)
    rows = []
    # Header: TransUnion, Experian, Equifax at x0 = 100, 300, 500
    rows.append((1, 1, 10.0, 12.0, 100.0, 140.0, "TransUnion"))
    rows.append((1, 1, 10.0, 12.0, 300.0, 340.0, "Experian"))
    rows.append((1, 1, 10.0, 12.0, 500.0, 540.0, "Equifax"))
    # Anchor below header
    rows.append((1, 2, 50.0, 52.0, 20.0, 60.0, "Account"))
    rows.append((1, 2, 50.0, 52.0, 80.0, 85.0, "#"))
    return rows


def test_no_eq_snap_left_of_eq_left(tmp_path: Path):
    # Token just left of eq_left_x0 must classify as XP (not EQ)
    tsv = tmp_path / "_debug_full.tsv"
    rows = _header_and_anchor_rows()
    # Labeled row: Account Status: (label at x0=60, 90), value near eq seam at x0=499.8
    rows.append((1, 3, 60.0, 62.0, 60.0, 80.0, "Account"))
    rows.append((1, 3, 60.0, 62.0, 90.0, 100.0, "Status:"))
    # Value just left of EQ seam -> should be XP
    rows.append((1, 3, 60.0, 62.0, 499.8, 505.0, "NEAR_EQ"))
    _emit(tsv, rows)

    data, _, _ = _run_split(tsv)
    acc = data["accounts"][0]
    tri = acc["triad_fields"]
    # Ensure XP got the token, EQ empty
    assert tri["experian"]["account_status"] == "NEAR_EQ"
    assert tri["equifax"]["account_status"] == ""


def test_label_stop_before_tu_left(tmp_path: Path):
    # Label collection must not swallow first TU value; value appears in TU
    tsv = tmp_path / "_debug_full.tsv"
    rows = _header_and_anchor_rows()
    # Labeled row: Account Status:
    rows.append((1, 3, 60.0, 62.0, 60.0, 80.0, "Account"))
    rows.append((1, 3, 60.0, 62.0, 90.0, 100.0, "Status:"))
    rows.append((1, 3, 60.0, 62.0, 120.0, 140.0, "ABC123"))
    _emit(tsv, rows)

    data, _, _ = _run_split(tsv)
    acc = data["accounts"][0]
    tri = acc["triad_fields"]
    # TU should capture the value; XP/EQ remain empty
    assert tri["transunion"]["account_status"] == "ABC123"
    assert tri["experian"]["account_status"] == ""
    assert tri["equifax"]["account_status"] == ""


def test_tu_long_wrap_keeps_all_words_strict(tmp_path: Path):
    # TU value continues on the next line; all words must be kept in TU only
    tsv = tmp_path / "_debug_full.tsv"
    rows = _header_and_anchor_rows()
    # Labeled line with TU tokens
    rows.append((1, 3, 60.0, 62.0, 60.0, 80.0, "Account"))
    rows.append((1, 3, 60.0, 62.0, 90.0, 100.0, "Status:"))
    rows.append((1, 3, 60.0, 62.0, 120.0, 140.0, "PAST"))
    rows.append((1, 3, 60.0, 62.0, 160.0, 180.0, "DUE"))
    # Continuation line: more TU tokens only
    rows.append((1, 4, 62.0, 64.0, 180.0, 200.0, "REPORTED"))
    rows.append((1, 4, 62.0, 64.0, 220.0, 240.0, "LATE"))
    _emit(tsv, rows)

    data, _, _ = _run_split(tsv)
    acc = data["accounts"][0]
    tri = acc["triad_fields"]
    assert tri["transunion"]["account_status"] == "PAST DUE REPORTED LATE"
    assert tri["experian"]["account_status"] == ""
    assert tri["equifax"]["account_status"] == ""


def test_eq_on_continuation_no_xp_bleed_strict(tmp_path: Path):
    # XP on labeled line; EQ on continuation. They must not bleed into each other.
    tsv = tmp_path / "_debug_full.tsv"
    rows = _header_and_anchor_rows()
    # Labeled line: XP token
    rows.append((1, 3, 60.0, 62.0, 60.0, 80.0, "Account"))
    rows.append((1, 3, 60.0, 62.0, 90.0, 100.0, "Status:"))
    rows.append((1, 3, 60.0, 62.0, 320.0, 340.0, "XPV"))
    # Continuation line: EQ token
    rows.append((1, 4, 62.0, 64.0, 520.0, 540.0, "EQVALUE"))
    _emit(tsv, rows)

    data, _, _ = _run_split(tsv)
    acc = data["accounts"][0]
    tri = acc["triad_fields"]
    assert tri["experian"]["account_status"] == "XPV"
    assert tri["equifax"]["account_status"] == "EQVALUE"
    # Ensure no cross-bureau contamination
    assert tri["transunion"]["account_status"] == ""
