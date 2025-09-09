import json
from pathlib import Path

from scripts import split_accounts_from_tsv


def create_sample_tsv(path: Path) -> None:
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


def test_split_accounts(tmp_path: Path) -> None:
    tsv_path = tmp_path / "_debug_full.tsv"
    json_path = tmp_path / "accounts_from_full.json"
    create_sample_tsv(tsv_path)

    split_accounts_from_tsv.main([
        "--full",
        str(tsv_path),
        "--json_out",
        str(json_path),
    ])

    data = json.loads(json_path.read_text())
    assert len(data) == 2

    acc1, acc2 = data
    assert acc1["heading_guess"] == "BANKAMERICA"
    assert acc1["line_start"] == 2
    assert acc1["line_end"] == 3
    assert acc1["lines"][0]["text"] == "Account # 123"

    assert acc2["heading_guess"] == "TRUISTMRTG"
    assert acc2["line_start"] == 5
    assert acc2["lines"][1]["text"] == "Line B1"
