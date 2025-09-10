import json
from pathlib import Path

from scripts import split_general_info_from_tsv


def create_basic_tsv(path: Path) -> None:
    content = (
        "page\tline\ty0\ty1\tx0\tx1\ttext\n"
        "1\t1\t0\t0\t0\t0\tPERSONAL INFORMATION\n"
        "1\t2\t0\t0\t0\t0\tName: JOHN DOE\n"
        "1\t3\t0\t0\t0\t0\tPUBLIC INFORMATION\n"
        "1\t4\t0\t0\t0\t0\tBankruptcy: None\n"
        "1\t5\t0\t0\t0\t0\tAccount # 123\n"
    )
    path.write_text(content, encoding="utf-8")


def test_basic_split(tmp_path: Path) -> None:
    tsv_path = tmp_path / "_debug_full.tsv"
    json_path = tmp_path / "general_info_from_full.json"
    create_basic_tsv(tsv_path)

    split_general_info_from_tsv.main(
        ["--full", str(tsv_path), "--json_out", str(json_path)]
    )

    data = json.loads(json_path.read_text())
    sections = data["sections"]
    assert len(sections) == 2

    sec1, sec2 = sections
    assert sec1["heading"] == "PERSONAL INFORMATION"
    assert [ln["text"] for ln in sec1["lines"]] == [
        "PERSONAL INFORMATION",
        "Name: JOHN DOE",
    ]
    assert sec2["heading"] == "PUBLIC INFORMATION"
    assert [ln["text"] for ln in sec2["lines"]] == [
        "PUBLIC INFORMATION",
        "Bankruptcy: None",
    ]


def create_no_section_tsv(path: Path) -> None:
    content = (
        "page\tline\ty0\ty1\tx0\tx1\ttext\n"
        "1\t1\t0\t0\t0\t0\tName: JOHN DOE\n"
        "1\t2\t0\t0\t0\t0\tAccount # 123\n"
    )
    path.write_text(content, encoding="utf-8")


def test_no_sections(tmp_path: Path) -> None:
    tsv_path = tmp_path / "_debug_full.tsv"
    json_path = tmp_path / "general_info_from_full.json"
    create_no_section_tsv(tsv_path)

    split_general_info_from_tsv.main(
        ["--full", str(tsv_path), "--json_out", str(json_path)]
    )

    data = json.loads(json_path.read_text())
    assert data["sections"] == []
