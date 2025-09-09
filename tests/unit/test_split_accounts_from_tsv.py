import json
from pathlib import Path

from scripts import split_accounts_from_tsv


def create_heading_backtrack_tsv(path: Path) -> None:
    content = (
        "page\tline\ty0\ty1\tx0\tx1\ttext\n"
        "1\t1\t0\t0\t0\t0\tBANKAMERICA\n"
        "1\t2\t0\t0\t0\t0\tFILLER\n"
        "1\t3\t0\t0\t0\t0\tAccount # 123\n"
        "1\t4\t0\t0\t0\t0\tLine A1\n"
        "1\t5\t0\t0\t0\t0\tTRUISTMRTG\n"
        "1\t6\t0\t0\t0\t0\tAccount # 456\n"
        "1\t7\t0\t0\t0\t0\tLine B1\n"
    )
    path.write_text(content, encoding="utf-8")


def test_heading_backtrack(tmp_path: Path) -> None:
    tsv_path = tmp_path / "_debug_full.tsv"
    json_path = tmp_path / "accounts_from_full.json"
    create_heading_backtrack_tsv(tsv_path)

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
    assert acc1["line_start"] == 1
    assert acc1["lines"][0]["text"] == "BANKAMERICA"

    assert acc2["heading_guess"] == "TRUISTMRTG"
    assert acc2["line_start"] == 5
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
