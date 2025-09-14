import json
import logging
import os
import runpy
from pathlib import Path


def _write_tsv(path: Path) -> None:
    # Build a minimal TSV with a header, an anchor row where XP has two tokens,
    # and a labeled High Balance row using a Unicode colon.
    header = "page\tline\ty0\ty1\tx0\tx1\ttext\n"
    rows = [
        # Page 1 triad header: TU/XP/EQ with midpoints ~80/180/280
        "1\t1\t10\t11\t60\t100\tTransUnion\n",
        "1\t1\t10\t11\t160\t200\tExperian\n",
        "1\t1\t10\t11\t260\t300\tEquifax\n",
        # Anchor row: label in label band; TU/EQ single token; XP split into two tokens
        "1\t2\t20\t21\t0\t20\tAccount #\n",
        "1\t2\t20\t21\t60\t100\t208743***\n",  # TU single token
        "1\t2\t20\t21\t160\t175\t208743\n",     # XP first token
        "1\t2\t20\t21\t175\t200\t***\n",         # XP second token
        "1\t2\t20\t21\t260\t300\t208743***\n",  # EQ single token
        # Labeled row with Unicode colon (full-width): High Balance：
        "1\t3\t30\t31\t0\t20\tHigh Balance：\n",
        "1\t3\t30\t31\t60\t100\t5000\n",
        "1\t3\t30\t31\t160\t200\t6000\n",
        "1\t3\t30\t31\t260\t300\t7000\n",
    ]
    path.write_text(header + "".join(rows), encoding="utf-8")


def test_anchor_with_multi_token_bureau_and_unicode_colon(tmp_path: Path, caplog):
    tsv_path = tmp_path / "_debug_full.tsv"
    json_path = tmp_path / "accounts_from_full.json"
    _write_tsv(tsv_path)

    # Configure environment to enable triad-from-x and space-joining
    os.environ["RAW_TRIAD_FROM_X"] = "1"
    os.environ["RAW_JOIN_TOKENS_WITH_SPACE"] = "1"
    os.environ["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])

    # Run the script's main under caplog to capture diagnostic output
    import scripts.split_accounts_from_tsv as mod  # type: ignore
    with caplog.at_level(logging.INFO):
        mod.main(["--full", str(tsv_path), "--json_out", str(json_path)])

    data = json.loads(json_path.read_text())
    acc = data["accounts"][0]

    # Assert no layout_mismatch stop on the anchor line
    assert "TRIAD_STOP reason=layout_mismatch" not in caplog.text

    # Check account numbers
    fields = acc["triad_fields"]
    assert fields["transunion"]["account_number_display"] == "208743***"
    assert fields["experian"]["account_number_display"]  # filled, value may contain a space
    assert fields["equifax"]["account_number_display"] == "208743***"

    # Check High Balance values (unicode colon recognized)
    assert fields["transunion"]["high_balance"] == "5000"
    assert fields["experian"]["high_balance"] == "6000"
    assert fields["equifax"]["high_balance"] == "7000"

    # Basic expected logs around anchor/header counts
    assert "TRIAD_ANCHOR_AT page=" in caplog.text
    assert "TRIAD_HEADER_ABOVE page=" in caplog.text
    assert "TRIAD_HEADER_XMIDS tu=" in caplog.text
    assert "TRIAD_ANCHOR_COUNTS label=1" in caplog.text


def _run_split(full: Path, out_json: Path, caplog):
    import scripts.split_accounts_from_tsv as mod  # type: ignore
    os.environ["RAW_TRIAD_FROM_X"] = "1"
    os.environ["RAW_JOIN_TOKENS_WITH_SPACE"] = "1"
    with caplog.at_level(logging.INFO):
        mod.main(["--full", str(full), "--json_out", str(out_json)])


def _csv_rows(path: Path) -> list[list[str]]:
    import csv
    with open(path, "r", encoding="utf-8") as fh:
        return list(csv.reader(fh))


def _write_no_label_anchor_with_h2y(tsv_path: Path) -> None:
    header = "page\tline\ty0\ty1\tx0\tx1\ttext\n"
    rows = [
        # Triad header
        "1\t1\t10\t11\t60\t100\tTransUnion\n",
        "1\t1\t10\t11\t160\t200\tExperian\n",
        "1\t1\t10\t11\t260\t300\tEquifax\n",
        # Anchor with label token placed in TU band to trigger fallback
        "1\t2\t20\t21\t72\t100\tAccount #\n",
        "1\t2\t20\t21\t100\t120\tTUANCH\n",
        "1\t2\t20\t21\t160\t200\tXPANCH\n",
        "1\t2\t20\t21\t260\t300\tEQANCH\n",
        # Labeled row
        "1\t3\t30\t31\t0\t20\tHigh Balance:\n",
        "1\t3\t30\t31\t60\t100\t5000\n",
        "1\t3\t30\t31\t160\t200\t6000\n",
        "1\t3\t30\t31\t260\t300\t7000\n",
        # Two-year payment history section
        "1\t4\t40\t41\t0\t50\tTwo Year Payment History\n",
        "1\t5\t40\t41\t160\t200\tExperian\n",
        "1\t6\t40\t41\t160\t170\tOK\n",
        "1\t6\t40\t41\t170\t180\t30\n",
        "1\t6\t40\t41\t180\t190\t60\n",
    ]
    tsv_path.write_text(header + "".join(rows), encoding="utf-8")


def _write_x0_anchor_and_header(tsv_path: Path,
                                tu_left: float,
                                xp_left: float,
                                eq_left: float,
                                extra_rows: list[str]) -> None:
    header = "page\tline\ty0\ty1\tx0\tx1\ttext\n"
    rows = [
        # Header with rough mids to establish bands
        "1\t1\t10\t11\t150\t200\tTransUnion\n",
        "1\t1\t10\t11\t290\t330\tExperian\n",
        "1\t1\t10\t11\t430\t480\tEquifax\n",
        # Anchor line: first TU/XP/EQ tokens define left-x0 cutoffs
        "1\t2\t20\t21\t0\t20\tAccount #\n",
        f"1\t2\t20\t21\t{tu_left}\t{tu_left+20}\tTUANCH\n",
        f"1\t2\t20\t21\t{xp_left}\t{xp_left+20}\tXPANCH\n",
        f"1\t2\t20\t21\t{eq_left}\t{eq_left+20}\tEQANCH\n",
    ]
    tsv_path.write_text(header + "".join(rows + extra_rows), encoding="utf-8")


def test_band_by_x0_three_values_single_line(tmp_path: Path, caplog):
    os.environ["TRIAD_BAND_BY_X0"] = "1"
    os.environ["TRIAD_TRACE_CSV"] = "1"
    tsv_path = tmp_path / "_debug_full.tsv"
    json_path = tmp_path / "accounts_from_full.json"
    tu_left, xp_left, eq_left = 172.875, 309.0, 445.125
    extra = [
        # Labeled row values almost at the left cutoffs
        "1\t3\t30\t31\t0\t10\tHigh Balance:\n",
        "1\t3\t30\t31\t172.9\t180\t$149,500\n",
        "1\t3\t30\t31\t309.0\t315\t$149,500\n",
        "1\t3\t30\t31\t445.1\t452\t$149,500\n",
    ]
    _write_x0_anchor_and_header(tsv_path, tu_left, xp_left, eq_left, extra)
    _run_split(tsv_path, json_path, caplog)
    data = json.loads(json_path.read_text())
    fields = data["accounts"][0]["triad_fields"]
    assert fields["transunion"]["high_balance"] == "$149,500"
    assert fields["experian"]["high_balance"] == "$149,500"
    assert fields["equifax"]["high_balance"] == "$149,500"
    # no TU rescue logs
    assert "TRIAD_TU_RESCUE" not in caplog.text
    # trace used_axis should be x0
    trace = tsv_path.parent / "per_account_tsv" / "_trace_account_1.csv"
    rows = _csv_rows(trace)
    header = rows[0]
    used_axis_idx = header.index("used_axis")
    assert all(r[used_axis_idx] == "x0" for r in rows[1:])


def test_term_length_tu_360_by_x0(tmp_path: Path, caplog):
    os.environ["TRIAD_BAND_BY_X0"] = "1"
    os.environ["TRIAD_TRACE_CSV"] = "1"
    tsv_path = tmp_path / "_debug_full.tsv"
    json_path = tmp_path / "accounts_from_full.json"
    tu_left, xp_left, eq_left = 172.875, 309.0, 445.125
    extra = [
        "1\t3\t30\t31\t0\t10\tTerm Length:\n",
        "1\t3\t30\t31\t172.9\t180\t360\n",
        "1\t3\t30\t31\t309.0\t315\t360\n",
        "1\t3\t30\t31\t445.1\t452\t360\n",
    ]
    _write_x0_anchor_and_header(tsv_path, tu_left, xp_left, eq_left, extra)
    _run_split(tsv_path, json_path, caplog)
    data = json.loads(json_path.read_text())
    fields = data["accounts"][0]["triad_fields"]
    assert fields["transunion"]["term_length"] == "360"


def test_tu_dash_rescue_by_x0(tmp_path: Path, caplog):
    os.environ["TRIAD_BAND_BY_X0"] = "1"
    os.environ["TRIAD_TRACE_CSV"] = "1"
    tsv_path = tmp_path / "_debug_full.tsv"
    json_path = tmp_path / "accounts_from_full.json"
    tu_left, xp_left, eq_left = 172.875, 309.0, 445.125
    extra = [
        "1\t3\t30\t31\t0\t10\tHigh Balance:\n",
        f"1\t3\t30\t31\t{tu_left - 1.0}\t{tu_left - 0.5}\t--\n",
    ]
    _write_x0_anchor_and_header(tsv_path, tu_left, xp_left, eq_left, extra)
    _run_split(tsv_path, json_path, caplog)
    data = json.loads(json_path.read_text())
    fields = data["accounts"][0]["triad_fields"]
    assert fields["transunion"]["high_balance"] == "--"
    assert "TRIAD_TU_RESCUE" in caplog.text


def test_wrapped_equifax_remarks_left_margin(tmp_path: Path, caplog):
    os.environ["TRIAD_BAND_BY_X0"] = "1"
    os.environ["TRIAD_TRACE_CSV"] = "1"
    tsv_path = tmp_path / "_debug_full.tsv"
    json_path = tmp_path / "accounts_from_full.json"
    tu_left, xp_left, eq_left = 172.875, 309.0, 445.125
    extra = [
        # Line 1 values: only Equifax gets text
        "1\t3\t30\t31\t0\t10\tCreditor Remarks:\n",
        "1\t3\t30\t31\t445.2\t455\tClosed\n",
        "1\t3\t30\t31\t460.0\t470\tor\n",
        "1\t3\t30\t31\t475.0\t485\tpaid\n",
        "1\t3\t30\t31\t490.0\t500\taccount/zero\n",
        # Line 2 continuation: left-of-TU tokens, should wrap to Equifax via carry_forward
        "1\t4\t40\t41\t10.0\t20\tFannie\n",
        "1\t4\t40\t41\t25.0\t35\tMae\n",
        "1\t4\t40\t41\t40.0\t55\taccount\n",
    ]
    _write_x0_anchor_and_header(tsv_path, tu_left, xp_left, eq_left, extra)
    _run_split(tsv_path, json_path, caplog)
    data = json.loads(json_path.read_text())
    fields = data["accounts"][0]["triad_fields"]
    val = fields["equifax"]["creditor_remarks"]
    assert val.endswith("Fannie Mae account")
    # Ensure no label-band tokens remain on continuation
    trace = tsv_path.parent / "per_account_tsv" / "_trace_account_1.csv"
    rows = _csv_rows(trace)
    header = rows[0]
    band_idx = header.index("band")
    phase_idx = header.index("phase")
    # no band=label with phase=cont
    assert not any(r[band_idx] == "label" and r[phase_idx] == "cont" for r in rows[1:])


def test_label_break_then_values_on_next_line(tmp_path: Path, caplog):
    os.environ["TRIAD_BAND_BY_X0"] = "1"
    os.environ["TRIAD_TRACE_CSV"] = "1"
    tsv_path = tmp_path / "_debug_full.tsv"
    json_path = tmp_path / "accounts_from_full.json"
    tu_left, xp_left, eq_left = 172.875, 309.0, 445.125
    extra = [
        # Label without suffix, ends before reaching TU x0
        "1\t3\t30\t31\t0\t10\tHigh Balance\n",
        # Next line contains the three values, first token left of TU
        "1\t4\t40\t41\t10.0\t20\t$1000\n",
        "1\t4\t40\t41\t309.0\t320\t$2000\n",
        "1\t4\t40\t41\t445.2\t455\t$3000\n",
    ]
    _write_x0_anchor_and_header(tsv_path, tu_left, xp_left, eq_left, extra)
    _run_split(tsv_path, json_path, caplog)
    data = json.loads(json_path.read_text())
    fields = data["accounts"][0]["triad_fields"]
    assert fields["transunion"]["high_balance"] == "$1000"
    assert fields["experian"]["high_balance"] == "$2000"
    assert fields["equifax"]["high_balance"] == "$3000"
    # Log should indicate we expected continuation
    assert "TRIAD_LABEL_LINEBREAK" in caplog.text


def test_balance_owed_wrap_tu_left_short_token(tmp_path: Path, caplog):
    os.environ["TRIAD_BAND_BY_X0"] = "1"
    os.environ["TRIAD_TRACE_CSV"] = "1"
    tsv_path = tmp_path / "_debug_full.tsv"
    json_path = tmp_path / "accounts_from_full.json"
    tu_left, xp_left, eq_left = 172.875, 309.0, 445.125
    extra = [
        # Line 1: TU has value; XP/EQ empty makes last_bureau TU
        "1\t3\t30\t31\t0\t10\tBalance Owed:\n",
        "1\t3\t30\t31\t172.9\t180\t$5\n",
        # Line 2 continuation: short token left of TU, should carry_forward to TU
        "1\t4\t40\t41\t10.0\t15\t$0\n",
    ]
    _write_x0_anchor_and_header(tsv_path, tu_left, xp_left, eq_left, extra)
    _run_split(tsv_path, json_path, caplog)
    data = json.loads(json_path.read_text())
    fields = data["accounts"][0]["triad_fields"]
    assert fields["transunion"]["balance_owed"].endswith("$0")
    # Continuation should not leave band=label entries
    trace = tsv_path.parent / "per_account_tsv" / "_trace_account_1.csv"
    rows = _csv_rows(trace)
    header = rows[0]
    band_idx = header.index("band")
    phase_idx = header.index("phase")
    assert not any(r[band_idx] == "label" and r[phase_idx] == "cont" for r in rows[1:])


def test_label_indented_in_label_band(tmp_path: Path, caplog):
    tsv_path = tmp_path / "_debug_full.tsv"
    json_path = tmp_path / "accounts_from_full.json"
    header = "page\tline\ty0\ty1\tx0\tx1\ttext\n"
    rows = [
        # Header with mids ~80/180/280
        "1\t1\t10\t11\t60\t100\tTransUnion\n",
        "1\t1\t10\t11\t160\t200\tExperian\n",
        "1\t1\t10\t11\t260\t300\tEquifax\n",
        # Anchor row
        "1\t2\t20\t21\t0\t20\tAccount #\n",
        "1\t2\t20\t21\t60\t100\tTU1\n",
        "1\t2\t20\t21\t160\t200\tXP1\n",
        "1\t2\t20\t21\t260\t300\tEQ1\n",
        # Labeled row with slight indent for label (x0=5,x1=10)
        "1\t3\t30\t31\t5\t10\tHigh Balance:\n",
        "1\t3\t30\t31\t60\t100\t5000\n",
        "1\t3\t30\t31\t160\t200\t6000\n",
        "1\t3\t30\t31\t260\t300\t7000\n",
    ]
    tsv_path.write_text(header + "".join(rows), encoding="utf-8")
    _run_split(tsv_path, json_path, caplog)

    data = json.loads(json_path.read_text())
    fields = data["accounts"][0]["triad_fields"]
    assert fields["transunion"]["high_balance"] == "5000"
    assert fields["experian"]["high_balance"] == "6000"
    assert fields["equifax"]["high_balance"] == "7000"


def test_seam_tie_break_right_to_xp(tmp_path: Path, caplog):
    tsv_path = tmp_path / "_debug_full.tsv"
    json_path = tmp_path / "accounts_from_full.json"
    header = "page\tline\ty0\ty1\tx0\tx1\ttext\n"
    rows = [
        # Header with TU mid 80, XP mid 180 -> b1 = 130
        "1\t1\t10\t11\t60\t100\tTransUnion\n",
        "1\t1\t10\t11\t160\t200\tExperian\n",
        "1\t1\t10\t11\t260\t300\tEquifax\n",
        # Anchor row
        "1\t2\t20\t21\t0\t20\tAccount #\n",
        "1\t2\t20\t21\t60\t100\tTU1\n",
        "1\t2\t20\t21\t160\t200\tXP1\n",
        "1\t2\t20\t21\t260\t300\tEQ1\n",
        # Labeled row: place XP value exactly on the TU/XP seam (mid=130)
        "1\t3\t30\t31\t0\t20\tHigh Balance:\n",
        "1\t3\t30\t31\t125\t135\tSEAM\n",
    ]
    tsv_path.write_text(header + "".join(rows), encoding="utf-8")
    _run_split(tsv_path, json_path, caplog)

    data = json.loads(json_path.read_text())
    fields = data["accounts"][0]["triad_fields"]
    assert fields["transunion"].get("high_balance", "") in {"", None}
    assert fields["experian"]["high_balance"] == "SEAM"
    assert fields["equifax"].get("high_balance", "") in {"", None}


def test_unknown_label_skipped_then_next_known_parsed(tmp_path: Path, caplog):
    tsv_path = tmp_path / "_debug_full.tsv"
    json_path = tmp_path / "accounts_from_full.json"
    header = "page\tline\ty0\ty1\tx0\tx1\ttext\n"
    rows = [
        # Header with mids ~80/180/280
        "1\t1\t10\t11\t60\t100\tTransUnion\n",
        "1\t1\t10\t11\t160\t200\tExperian\n",
        "1\t1\t10\t11\t260\t300\tEquifax\n",
        # Anchor row
        "1\t2\t20\t21\t0\t20\tAccount #\n",
        "1\t2\t20\t21\t60\t100\tTU1\n",
        "1\t2\t20\t21\t160\t200\tXP1\n",
        "1\t2\t20\t21\t260\t300\tEQ1\n",
        # Unknown label row
        "1\t3\t30\t31\t0\t20\tMystery Label:\n",
        "1\t3\t30\t31\t60\t100\tIGNORED\n",
        # Next known labeled row
        "1\t4\t40\t41\t0\t20\tHigh Balance:\n",
        "1\t4\t40\t41\t60\t100\t5000\n",
        "1\t4\t40\t41\t160\t200\t6000\n",
        "1\t4\t40\t41\t260\t300\t7000\n",
    ]
    tsv_path.write_text(header + "".join(rows), encoding="utf-8")
    _run_split(tsv_path, json_path, caplog)

    data = json.loads(json_path.read_text())
    fields = data["accounts"][0]["triad_fields"]
    assert fields["transunion"]["high_balance"] == "5000"
    assert fields["experian"]["high_balance"] == "6000"
    assert fields["equifax"]["high_balance"] == "7000"
    assert "TRIAD_GUARD_SKIP reason=unknown_label" in caplog.text

def test_multitoken_high_balance_label_builds_and_maps(tmp_path: Path, caplog):
    tsv_path = tmp_path / "_debug_full.tsv"
    json_path = tmp_path / "accounts_from_full.json"
    trace_dir = tmp_path / "per_account_tsv"
    os.environ["TRIAD_TRACE_CSV"] = "1"

    header = "page\tline\ty0\ty1\tx0\tx1\ttext\n"
    rows = [
        # Header with mids ~80/180/280
        "1\t1\t10\t11\t60\t100\tTransUnion\n",
        "1\t1\t10\t11\t160\t200\tExperian\n",
        "1\t1\t10\t11\t260\t300\tEquifax\n",
        # Anchor row
        "1\t2\t20\t21\t0\t20\tAccount\n",
        "1\t2\t20\t21\t20\t30\t#\n",
        "1\t2\t20\t21\t60\t100\tTU1\n",
        "1\t2\t20\t21\t160\t200\tXP1\n",
        "1\t2\t20\t21\t260\t300\tEQ1\n",
        # Multi-token High Balance label: "High" + "Balance:"
        "1\t3\t30\t31\t0\t10\tHigh\n",
        "1\t3\t30\t31\t10\t30\tBalance:\n",
        "1\t3\t30\t31\t60\t100\t5000\n",
        "1\t3\t30\t31\t160\t200\t6000\n",
        "1\t3\t30\t31\t260\t300\t7000\n",
    ]
    tsv_path.write_text(header + "".join(rows), encoding="utf-8")
    _run_split(tsv_path, json_path, caplog)

    data = json.loads(json_path.read_text())
    fields = data["accounts"][0]["triad_fields"]
    assert fields["transunion"]["high_balance"] == "5000"
    assert fields["experian"]["high_balance"] == "6000"
    assert fields["equifax"]["high_balance"] == "7000"
    # Trace: ensure label_key is resolved for labeled tokens
    trace_files = list(trace_dir.glob("_trace_account_*.csv"))
    assert trace_files, "trace CSV not found"
    import csv as _csv
    with trace_files[0].open("r", encoding="utf-8", newline="") as fh:
        rd = _csv.DictReader(fh)
        labeled_labels = [r for r in rd if r.get("phase") == "labeled" and r.get("band") == "label" and r.get("line") == '3']
    assert labeled_labels, "no labeled label tokens traced"
    # All label tokens for this row should map to high_balance
    assert all((r.get("label_key") == "high_balance") for r in labeled_labels)


def test_multitoken_account_hash_label_builds_and_maps(tmp_path: Path, caplog):
    tsv_path = tmp_path / "_debug_full.tsv"
    json_path = tmp_path / "accounts_from_full.json"
    trace_dir = tmp_path / "per_account_tsv"
    os.environ["TRIAD_TRACE_CSV"] = "1"

    header = "page\tline\ty0\ty1\tx0\tx1\ttext\n"
    rows = [
        # Header with mids ~80/180/280
        "1\t1\t10\t11\t60\t100\tTransUnion\n",
        "1\t1\t10\t11\t160\t200\tExperian\n",
        "1\t1\t10\t11\t260\t300\tEquifax\n",
        # Anchor label split across two tokens: "Account" + "#"
        "1\t2\t20\t21\t0\t20\tAccount\n",
        "1\t2\t20\t21\t20\t30\t#\n",
        # Values
        "1\t2\t20\t21\t60\t100\tTU999\n",
        "1\t2\t20\t21\t160\t200\tXP999\n",
        "1\t2\t20\t21\t260\t300\tEQ999\n",
    ]
    tsv_path.write_text(header + "".join(rows), encoding="utf-8")
    _run_split(tsv_path, json_path, caplog)

    data = json.loads(json_path.read_text())
    fields = data["accounts"][0]["triad_fields"]
    assert fields["transunion"]["account_number_display"] == "TU999"
    assert fields["experian"]["account_number_display"] == "XP999"
    assert fields["equifax"]["account_number_display"] == "EQ999"
    # Trace
    import csv as _csv
    trace_files = list(trace_dir.glob("_trace_account_*.csv"))
    assert trace_files, "trace CSV not found"
    with trace_files[0].open("r", encoding="utf-8", newline="") as fh:
        rd = _csv.DictReader(fh)
        labeled_labels = [r for r in rd if r.get("phase") == "labeled" and r.get("band") == "label" and r.get("line") == '2']
    assert labeled_labels, "no labeled label tokens traced"
    assert all((r.get("label_key") == "account_number_display") for r in labeled_labels)

def test_labeled_row_missing_bureau_value_is_kept(tmp_path: Path, caplog):
    tsv_path = tmp_path / "_debug_full.tsv"
    json_path = tmp_path / "accounts_from_full.json"

    header = "page\tline\ty0\ty1\tx0\tx1\ttext\n"
    rows = [
        # Header with mids ~80/180/280
        "1\t1\t10\t11\t60\t100\tTransUnion\n",
        "1\t1\t10\t11\t160\t200\tExperian\n",
        "1\t1\t10\t11\t260\t300\tEquifax\n",
        # Anchor row
        "1\t2\t20\t21\t0\t20\tAccount #\n",
        "1\t2\t20\t21\t60\t100\tTU1\n",
        "1\t2\t20\t21\t160\t200\tXP1\n",
        "1\t2\t20\t21\t260\t300\tEQ1\n",
        # Labeled row missing XP value: only TU and EQ provided
        "1\t3\t30\t31\t0\t20\tHigh Balance:\n",
        "1\t3\t30\t31\t60\t100\t5000\n",
        "1\t3\t30\t31\t260\t300\t7000\n",
    ]
    tsv_path.write_text(header + "".join(rows), encoding="utf-8")
    _run_split(tsv_path, json_path, caplog)

    data = json.loads(json_path.read_text())
    acc = data["accounts"][0]
    fields = acc["triad_fields"]
    # triad_fields updated where present
    assert fields["transunion"]["high_balance"] == "5000"
    assert fields["equifax"]["high_balance"] == "7000"
    # experian may be missing the key or be empty string; ensure triad_rows captured empty value
    row = next(r for r in acc["triad_rows"] if r.get("key") == "high_balance")
    assert row["values"]["transunion"] == "5000"
    assert row["values"]["experian"] == ""
    assert row["values"]["equifax"] == "7000"


def test_tu_value_near_label_rescued_high_balance(tmp_path: Path, caplog):
    tsv_path = tmp_path / "_debug_full.tsv"
    json_path = tmp_path / "accounts_from_full.json"

    header = "page\tline\ty0\ty1\tx0\tx1\ttext\n"
    rows = [
        # Header with mids ~80/180/280
        "1\t1\t10\t11\t60\t100\tTransUnion\n",
        "1\t1\t10\t11\t160\t200\tExperian\n",
        "1\t1\t10\t11\t260\t300\tEquifax\n",
        # Anchor row
        "1\t2\t20\t21\t0\t20\tAccount #\n",
        "1\t2\t20\t21\t60\t100\tTU1\n",
        "1\t2\t20\t21\t160\t200\tXP1\n",
        "1\t2\t20\t21\t260\t300\tEQ1\n",
        # High Balance with TU mis-banded into label (immediately after label)
        "1\t3\t30\t31\t0\t20\tHigh Balance:\n",
        "1\t3\t30\t31\t5\t25\t$5000\n",     # falls in label band, should be rescued to TU
        "1\t3\t30\t31\t160\t200\t$6000\n",
        "1\t3\t30\t31\t260\t300\t$7000\n",
    ]
    tsv_path.write_text(header + "".join(rows), encoding="utf-8")
    _run_split(tsv_path, json_path, caplog)

    data = json.loads(json_path.read_text())
    fields = data["accounts"][0]["triad_fields"]
    assert fields["transunion"]["high_balance"] == "$5000"
    assert fields["experian"]["high_balance"] == "$6000"
    assert fields["equifax"]["high_balance"] == "$7000"


def test_tu_360_near_label_rescued_term_length(tmp_path: Path, caplog):
    tsv_path = tmp_path / "_debug_full.tsv"
    json_path = tmp_path / "accounts_from_full.json"

    header = "page\tline\ty0\ty1\tx0\tx1\ttext\n"
    rows = [
        # Header
        "1\t1\t10\t11\t60\t100\tTransUnion\n",
        "1\t1\t10\t11\t160\t200\tExperian\n",
        "1\t1\t10\t11\t260\t300\tEquifax\n",
        # Anchor
        "1\t2\t20\t21\t0\t20\tAccount #\n",
        "1\t2\t20\t21\t60\t100\tTU1\n",
        "1\t2\t20\t21\t160\t200\tXP1\n",
        "1\t2\t20\t21\t260\t300\tEQ1\n",
        # Term Length with TU numeric near TU seam but in label band; XP/EQ have values
        "1\t3\t30\t31\t0\t20\tTerm Length:\n",
        "1\t3\t30\t31\t70\t72\t360\n",   # midpoint ~71, likely within rescue window
        "1\t3\t30\t31\t160\t180\t720\n",
        "1\t3\t30\t31\t260\t280\t540\n",
    ]
    tsv_path.write_text(header + "".join(rows), encoding="utf-8")
    _run_split(tsv_path, json_path, caplog)

    data = json.loads(json_path.read_text())
    fields = data["accounts"][0]["triad_fields"]
    assert fields["transunion"]["term_length"] == "360"
    assert fields["experian"]["term_length"] == "720"
    assert fields["equifax"]["term_length"] == "540"


def test_creditor_type_left_tokens_rescued(tmp_path: Path, caplog):
    tsv_path = tmp_path / "_debug_full.tsv"
    json_path = tmp_path / "accounts_from_full.json"

    header = "page\tline\ty0\ty1\tx0\tx1\ttext\n"
    rows = [
        # Header
        "1\t1\t10\t11\t60\t100\tTransUnion\n",
        "1\t1\t10\t11\t160\t200\tExperian\n",
        "1\t1\t10\t11\t260\t300\tEquifax\n",
        # Anchor
        "1\t2\t20\t21\t0\t20\tAccount #\n",
        "1\t2\t20\t21\t60\t100\tTU1\n",
        "1\t2\t20\t21\t160\t200\tXP1\n",
        "1\t2\t20\t21\t260\t300\tEQ1\n",
        # Creditor Type with TU words placed near left but inside TU band so geometry takes it
        "1\t3\t30\t31\t0\t20\tCreditor Type:\n",
        "1\t3\t30\t31\t72\t88\tBank\n",   # midpoint ~80 places inside TU band
        "1\t3\t30\t31\t160\t200\tFinance\n",
        "1\t3\t30\t31\t260\t300\tMortgage\n",
    ]
    tsv_path.write_text(header + "".join(rows), encoding="utf-8")
    _run_split(tsv_path, json_path, caplog)

    data = json.loads(json_path.read_text())
    fields = data["accounts"][0]["triad_fields"]
    assert fields["transunion"]["creditor_type"] == "Bank"
    assert fields["experian"]["creditor_type"] == "Finance"
    assert fields["equifax"]["creditor_type"] == "Mortgage"


def test_triad_x0_fallback_and_h2y_bureau_line(tmp_path: Path, caplog):
    os.environ["TRIAD_BAND_BY_X0"] = "1"
    os.environ["TRIAD_TRACE_CSV"] = "1"
    os.environ["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
    tsv_path = tmp_path / "_debug_full.tsv"
    json_path = tmp_path / "accounts_from_full.json"
    _write_no_label_anchor_with_h2y(tsv_path)
    _run_split(tsv_path, json_path, caplog)
    data = json.loads(json_path.read_text())
    acc = data["accounts"][0]
    fields = acc["triad_fields"]
    assert fields["experian"]["high_balance"] == "6000"
    assert acc["two_year_payment_history"]["experian"] == ["OK", "30", "60"]
    assert "TRIAD_X0_FALLBACK_OK" in caplog.text
    assert "layout_mismatch_anchor" not in caplog.text
    trace = tsv_path.parent / "per_account_tsv" / "_trace_account_1.csv"
    rows = _csv_rows(trace)
    header = rows[0]
    phase_idx = header.index("phase")
    band_idx = header.index("band")
    assert any(r[phase_idx] == "labeled" for r in rows[1:])
    assert any(r[phase_idx] == "history2y" and r[band_idx] == "XP" for r in rows[1:])
