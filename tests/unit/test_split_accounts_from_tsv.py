import json
from pathlib import Path

from scripts import split_accounts_from_tsv


def create_all_caps_backtrack_tsv(path: Path) -> None:
    content = (
        "page\tline\ty0\ty1\tx0\tx1\ttext\n"
        "1\t1\t0\t0\t0\t0\tBANKAMERICA\n"
        "1\t2\t0\t0\t0\t0\tAccount # 123\n"
        "1\t3\t0\t0\t0\t0\tLine A1\n"
        "1\t4\t0\t0\t0\t0\tTRUISTMRTG\n"
        "1\t5\t0\t0\t0\t0\tAccount # 456\n"
        "1\t6\t0\t0\t0\t0\tLine B1\n"
    )
    path.write_text(content, encoding="utf-8")


def test_all_caps_backtrack(tmp_path: Path) -> None:
    tsv_path = tmp_path / "_debug_full.tsv"
    json_path = tmp_path / "accounts_from_full.json"
    create_all_caps_backtrack_tsv(tsv_path)

    split_accounts_from_tsv.main(
        [
            "--full",
            str(tsv_path),
            "--json_out",
            str(json_path),
        ]
    )

    data = json.loads(json_path.read_text())
    accounts = data["accounts"]
    assert len(accounts) == 2
    assert data["stop_marker_seen"] is False

    acc1, acc2 = accounts
    assert acc1["heading_guess"] == "BANKAMERICA"
    assert acc1["heading_source"] == "backtrack"
    assert acc1["line_start"] == 1
    assert acc1["lines"][0]["text"] == "BANKAMERICA"

    assert acc2["heading_guess"] == "TRUISTMRTG"
    assert acc2["heading_source"] == "backtrack"
    assert acc2["line_start"] == 4
    assert acc2["lines"][1]["text"] == "Account # 456"


def create_stop_marker_tsv(path: Path) -> None:
    content = (
        "page\tline\ty0\ty1\tx0\tx1\ttext\n"
        "1\t1\t0\t0\t0\t0\tBANKAMERICA\n"
        "1\t2\t0\t0\t0\t0\tAccount # 123\n"
        "1\t3\t0\t0\t0\t0\tLine A1\n"
        "1\t4\t0\t0\t0\t0\tPublic Information\n"
        "1\t5\t0\t0\t0\t0\tAccount # 999\n"
    )
    path.write_text(content, encoding="utf-8")


def test_stop_marker(tmp_path: Path) -> None:
    tsv_path = tmp_path / "_debug_full.tsv"
    json_path = tmp_path / "accounts_from_full.json"
    create_stop_marker_tsv(tsv_path)

    split_accounts_from_tsv.main(
        [
            "--full",
            str(tsv_path),
            "--json_out",
            str(json_path),
        ]
    )

    data = json.loads(json_path.read_text())
    accounts = data["accounts"]
    assert data["stop_marker_seen"] is True
    assert len(accounts) == 1
    acc = accounts[0]
    assert acc["line_end"] == 3
    texts = [ln["text"] for ln in acc["lines"]]
    assert all("Public Information" not in t for t in texts)


def create_collection_tsv(path: Path) -> None:
    content = (
        "page\tline\ty0\ty1\tx0\tx1\ttext\n"
        "1\t1\t0\t0\t0\t0\tAMEX\n"
        "1\t2\t0\t0\t0\t0\tAccount # 123\n"
        "1\t3\t0\t0\t0\t0\tLine A1\n"
        "1\t4\t0\t0\t0\t0\tCollection\n"
        "1\t5\t0\t0\t0\t0\tCREDENCE RM\n"
        "1\t6\t0\t0\t0\t0\tAccount # 456\n"
        "1\t7\t0\t0\t0\t0\tLine B1\n"
    )
    path.write_text(content, encoding="utf-8")


def test_collection_starts_new_account(tmp_path: Path) -> None:
    tsv_path = tmp_path / "_debug_full.tsv"
    json_path = tmp_path / "accounts_from_full.json"
    create_collection_tsv(tsv_path)

    split_accounts_from_tsv.main(
        ["--full", str(tsv_path), "--json_out", str(json_path)]
    )

    data = json.loads(json_path.read_text())
    accounts = data["accounts"]
    assert len(accounts) == 2
    acc1, acc2 = accounts
    texts1 = [ln["text"] for ln in acc1["lines"]]
    assert "Collection" not in texts1
    assert acc2["heading_guess"] == "CREDENCE RM"
    assert acc2["section"] == "collections"


def create_unknown_tsv(path: Path) -> None:
    content = (
        "page\tline\ty0\ty1\tx0\tx1\ttext\n"
        "1\t1\t0\t0\t0\t0\tAMEX\n"
        "1\t2\t0\t0\t0\t0\tAccount # 123\n"
        "1\t3\t0\t0\t0\t0\tLine A1\n"
        "1\t4\t0\t0\t0\t0\tUnknown\n"
        "1\t5\t0\t0\t0\t0\tCREDITOR XYZ\n"
        "1\t6\t0\t0\t0\t0\tAccount # 456\n"
        "1\t7\t0\t0\t0\t0\tLine B1\n"
    )
    path.write_text(content, encoding="utf-8")


def test_unknown_starts_new_account(tmp_path: Path) -> None:
    tsv_path = tmp_path / "_debug_full.tsv"
    json_path = tmp_path / "accounts_from_full.json"
    create_unknown_tsv(tsv_path)

    split_accounts_from_tsv.main(["--full", str(tsv_path), "--json_out", str(json_path)])

    data = json.loads(json_path.read_text())
    accounts = data["accounts"]
    assert len(accounts) == 2
    acc1, acc2 = accounts
    texts1 = [ln["text"] for ln in acc1["lines"]]
    assert "Unknown" not in texts1
    assert acc2["section"] == "unknown"
    assert acc2["heading_source"] == "section+heading"
    assert acc2["heading_guess"] == "CREDITOR XYZ"


def create_section_forward_scan_tsv(path: Path) -> None:
    content = (
        "page\tline\ty0\ty1\tx0\tx1\ttext\n"
        "1\t1\t0\t0\t0\t0\tCollection\n"
        "1\t2\t0\t0\t0\t0\tTransunion®Experian®Equifax®\n"
        "1\t3\t0\t0\t0\t0\tCREDENCE RM\n"
        "1\t4\t0\t0\t0\t0\tAccount # 123\n"
    )
    path.write_text(content, encoding="utf-8")


def test_heading_from_section_forward_scan(tmp_path: Path) -> None:
    tsv_path = tmp_path / "_debug_full.tsv"
    json_path = tmp_path / "accounts_from_full.json"
    create_section_forward_scan_tsv(tsv_path)

    split_accounts_from_tsv.main(["--full", str(tsv_path), "--json_out", str(json_path)])

    data = json.loads(json_path.read_text())
    accounts = data["accounts"]
    assert len(accounts) == 1
    acc = accounts[0]
    assert acc["heading_guess"] == "CREDENCE RM"
    assert acc["line_start"] == 3
    assert acc["heading_source"] == "section+heading"


def create_trailing_collection_tsv(path: Path) -> None:
    content = (
        "page\tline\ty0\ty1\tx0\tx1\ttext\n"
        "1\t1\t0\t0\t0\t0\tAMEX\n"
        "1\t2\t0\t0\t0\t0\tAccount # 123\n"
        "1\t3\t0\t0\t0\t0\tLine A1\n"
        "1\t4\t0\t0\t0\t0\tCollection\n"
    )
    path.write_text(content, encoding="utf-8")


def test_trailing_section_marker_never_in_account(tmp_path: Path) -> None:
    tsv_path = tmp_path / "_debug_full.tsv"
    json_path = tmp_path / "accounts_from_full.json"
    create_trailing_collection_tsv(tsv_path)

    split_accounts_from_tsv.main(["--full", str(tsv_path), "--json_out", str(json_path)])

    data = json.loads(json_path.read_text())
    accounts = data["accounts"]
    assert len(accounts) == 1
    acc = accounts[0]
    texts = [ln["text"] for ln in acc["lines"]]
    assert "Collection" not in texts
    assert acc["trailing_section_marker_pruned"] is True


def create_section_prefix_end_tsv(path: Path) -> None:
    content = (
        "page\tline\ty0\ty1\tx0\tx1\ttext\n"
        "1\t1\t0\t0\t0\t0\tAMEX\n"
        "1\t2\t0\t0\t0\t0\tAccount # 123\n"
        "1\t3\t0\t0\t0\t0\tLine A1\n"
        "1\t4\t0\t0\t0\t0\tCollection\n"
        "1\t5\t0\t0\t0\t0\tCREDENCE RM\n"
        "1\t6\t0\t0\t0\t0\tAccount # 456\n"
        "1\t7\t0\t0\t0\t0\tLine B1\n"
        "1\t8\t0\t0\t0\t0\tUnknown\n"
    )
    path.write_text(content, encoding="utf-8")


def test_no_account_ends_with_section_prefix(tmp_path: Path) -> None:
    tsv_path = tmp_path / "_debug_full.tsv"
    json_path = tmp_path / "accounts_from_full.json"
    create_section_prefix_end_tsv(tsv_path)

    split_accounts_from_tsv.main(["--full", str(tsv_path), "--json_out", str(json_path)])

    data = json.loads(json_path.read_text())
    accounts = data["accounts"]
    assert len(accounts) == 2
    for acc in accounts:
        assert (
            split_accounts_from_tsv._norm(acc["lines"][-1]["text"])
            not in split_accounts_from_tsv.SECTION_STARTERS
        )
    assert accounts[0]["trailing_section_marker_pruned"] is True
    assert accounts[1]["trailing_section_marker_pruned"] is True


def create_noise_between_accounts_tsv(path: Path) -> None:
    content = (
        "page\tline\ty0\ty1\tx0\tx1\ttext\n"
        "1\t1\t0\t0\t0\t0\tAMEX\n"
        "1\t2\t0\t0\t0\t0\tAccount # 123\n"
        "1\t3\t0\t0\t0\t0\tLine A1\n"
        "1\t4\t0\t0\t0\t0\thttps://example.com/3b?foo\n"
        "1\t5\t0\t0\t0\t0\t5/28/25, 12:47 PM 3-Bureau Credit Report\n"
        "1\t6\t0\t0\t0\t0\tCollection\n"
        "1\t7\t0\t0\t0\t0\tCREDENCE RM\n"
        "1\t8\t0\t0\t0\t0\tAccount # 456\n"
        "1\t9\t0\t0\t0\t0\tLine B1\n"
    )
    path.write_text(content, encoding="utf-8")


def test_noise_lines_skipped(tmp_path: Path) -> None:
    tsv_path = tmp_path / "_debug_full.tsv"
    json_path = tmp_path / "accounts_from_full.json"
    create_noise_between_accounts_tsv(tsv_path)

    split_accounts_from_tsv.main(["--full", str(tsv_path), "--json_out", str(json_path)])

    data = json.loads(json_path.read_text())
    accounts = data["accounts"]
    assert len(accounts) == 2
    acc1, acc2 = accounts
    texts1 = [ln["text"] for ln in acc1["lines"]]
    assert "https://example.com/3b?foo" not in texts1
    assert "5/28/25, 12:47 PM 3-Bureau Credit Report" not in texts1
    assert acc1["noise_lines_skipped"] == 2
    assert acc2["noise_lines_skipped"] == 0


def create_triad_above_tsv(path: Path) -> None:
    content = (
        "page\tline\ty0\ty1\tx0\tx1\ttext\n"
        "1\t1\t0\t0\t0\t0\tCREDITOR XYZ\n"
        "1\t2\t0\t0\t0\t0\tTransunion®Experian®Equifax®\n"
        "1\t3\t0\t0\t0\t0\tAccount # 123\n"
        "1\t4\t0\t0\t0\t0\tLine A1\n"
    )
    path.write_text(content, encoding="utf-8")


def test_triad_above_selection(tmp_path: Path) -> None:
    tsv_path = tmp_path / "_debug_full.tsv"
    json_path = tmp_path / "accounts_from_full.json"
    create_triad_above_tsv(tsv_path)

    split_accounts_from_tsv.main(["--full", str(tsv_path), "--json_out", str(json_path)])

    data = json.loads(json_path.read_text())
    accounts = data["accounts"]
    assert len(accounts) == 1
    acc = accounts[0]
    assert acc["heading_guess"] == "CREDITOR XYZ"
    assert acc["heading_source"] == "triad_above"


def create_two_above_account_hash_tsv(path: Path) -> None:
    content = (
        "page\tline\ty0\ty1\tx0\tx1\ttext\n"
        "1\t1\t0\t0\t0\t0\tCREDITOR ABC\n"
        "1\t2\t0\t0\t0\t0\tfiller\n"
        "1\t3\t0\t0\t0\t0\tAccount # 999\n"
        "1\t4\t0\t0\t0\t0\tLine A1\n"
    )
    path.write_text(content, encoding="utf-8")


def test_two_above_account_hash_headline_rule(tmp_path: Path) -> None:
    tsv_path = tmp_path / "_debug_full.tsv"
    json_path = tmp_path / "accounts_from_full.json"
    create_two_above_account_hash_tsv(tsv_path)

    split_accounts_from_tsv.main(["--full", str(tsv_path), "--json_out", str(json_path)])

    data = json.loads(json_path.read_text())
    accounts = data["accounts"]
    assert len(accounts) == 1
    acc = accounts[0]
    assert acc["heading_guess"] == "CREDITOR ABC"
    assert acc["heading_source"] == "backtrack"


def create_anchor_only_tsv(path: Path) -> None:
    content = (
        "page\tline\ty0\ty1\tx0\tx1\ttext\n"
        "1\t1\t0\t0\t0\t0\tAccount # 123\n"
        "1\t2\t0\t0\t0\t0\tLine A1\n"
    )
    path.write_text(content, encoding="utf-8")


def test_anchor_not_used_as_headline(tmp_path: Path) -> None:
    tsv_path = tmp_path / "_debug_full.tsv"
    json_path = tmp_path / "accounts_from_full.json"
    create_anchor_only_tsv(tsv_path)

    split_accounts_from_tsv.main(["--full", str(tsv_path), "--json_out", str(json_path)])

    data = json.loads(json_path.read_text())
    accounts = data["accounts"]
    assert len(accounts) == 1
    acc = accounts[0]
    assert acc["heading_guess"] is None
    assert acc["heading_source"] == "anchor_no_heading"


def create_plain_account_tsv(path: Path) -> None:
    content = (
        "page\tline\ty0\ty1\tx0\tx1\ttext\n"
        "1\t1\t0\t0\t0\t0\tREAL BANK\n"
        "1\t2\t0\t0\t0\t0\tAccount Status:\n"
        "1\t3\t0\t0\t0\t0\tAccount # 123\n"
        "1\t4\t0\t0\t0\t0\tLine A1\n"
    )
    path.write_text(content, encoding="utf-8")


def test_plain_account_not_anchor(tmp_path: Path) -> None:
    tsv_path = tmp_path / "_debug_full.tsv"
    json_path = tmp_path / "accounts_from_full.json"
    create_plain_account_tsv(tsv_path)

    split_accounts_from_tsv.main(["--full", str(tsv_path), "--json_out", str(json_path)])

    data = json.loads(json_path.read_text())
    accounts = data["accounts"]
    assert len(accounts) == 1
    acc = accounts[0]
    assert acc["heading_guess"] == "REAL BANK"
    assert acc["heading_source"] == "backtrack"


def create_triad_moved_tsv(path: Path) -> None:
    content = (
        "page\tline\ty0\ty1\tx0\tx1\ttext\n"
        "1\t1\t0\t0\t0\t0\tBANK ONE\n"
        "1\t2\t0\t0\t0\t0\tAccount # 111\n"
        "1\t3\t0\t0\t0\t0\tLine A1\n"
        "1\t4\t0\t0\t0\t0\tTransUnion Experian Equifax\n"
        "1\t5\t0\t0\t0\t0\tCREDITOR TWO\n"
        "1\t6\t0\t0\t0\t0\tAccount # 222\n"
        "1\t7\t0\t0\t0\t0\tLine B1\n"
    )
    path.write_text(content, encoding="utf-8")


def test_triad_moved_to_next_account(tmp_path: Path) -> None:
    tsv_path = tmp_path / "_debug_full.tsv"
    json_path = tmp_path / "accounts_from_full.json"
    create_triad_moved_tsv(tsv_path)

    split_accounts_from_tsv.main(["--full", str(tsv_path), "--json_out", str(json_path)])

    data = json.loads(json_path.read_text())
    accounts = data["accounts"]
    assert len(accounts) == 2
    acc1, acc2 = accounts
    texts1 = [ln["text"] for ln in acc1["lines"]]
    assert "TransUnion Experian Equifax" not in texts1
    assert acc2["lines"][0]["text"] == "TransUnion Experian Equifax"


def create_h7y_tsv(path: Path) -> None:
    content = (
        "page\tline\ty0\ty1\tx0\tx1\ttext\n"
        "1\t1\t0\t0\t0\t0\tBANK\n"
        "1\t2\t0\t0\t0\t0\tAccount # 123\n"
        "1\t3\t0\t0\t0\t0\tDays Late - 7 Year History\n"
        "1\t4\t0\t0\t10\t20\tTransunion\n"
        "1\t4\t0\t0\t50\t60\tExperian\n"
        "1\t4\t0\t0\t90\t100\tEquifax\n"
        "1\t5\t0\t0\t10\t20\t30:\n"
        "1\t5\t0\t0\t20\t25\t--\n"
        "1\t5\t0\t0\t30\t40\t60:\n"
        "1\t5\t0\t0\t40\t45\t--\n"
        "1\t5\t0\t0\t44\t54\t90:\n"
        "1\t5\t0\t0\t54\t55\t--\n"
        "1\t5\t0\t0\t60\t70\t30:1\n"
        "1\t5\t0\t0\t70\t80\t60:1\n"
        "1\t5\t0\t0\t80\t90\t90:3\n"
        "1\t5\t0\t0\t100\t110\t30:\n"
        "1\t5\t0\t0\t110\t115\t--\n"
        "1\t5\t0\t0\t115\t125\t60:\n"
        "1\t5\t0\t0\t125\t130\t--\n"
        "1\t5\t0\t0\t130\t140\t90:\n"
        "1\t5\t0\t0\t140\t145\t--\n"
    )
    path.write_text(content, encoding="utf-8")


def test_seven_year_history_parsing(tmp_path: Path) -> None:
    tsv_path = tmp_path / "_debug_full.tsv"
    json_path = tmp_path / "accounts_from_full.json"
    create_h7y_tsv(tsv_path)

    split_accounts_from_tsv.main(["--full", str(tsv_path), "--json_out", str(json_path)])

    data = json.loads(json_path.read_text())
    accounts = data["accounts"]
    assert len(accounts) == 1
    sev = accounts[0]["seven_year_history"]
    assert sev["transunion"] == {"late30": 0, "late60": 0, "late90": 0}
    assert sev["experian"] == {"late30": 1, "late60": 1, "late90": 3}
    assert sev["equifax"] == {"late30": 0, "late60": 0, "late90": 0}
