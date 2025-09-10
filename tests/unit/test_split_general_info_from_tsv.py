import json
from pathlib import Path

from scripts import split_general_info_from_tsv


def create_full_tsv(path: Path) -> None:
    content = (
        "page\tline\ty0\ty1\tx0\tx1\ttext\n"
        "1\t1\t0\t0\t0\t0\tPERSONAL INFORMATION\n"
        "1\t2\t0\t0\t0\t0\tName: JOHN DOE\n"
        "1\t3\t0\t0\t0\t0\tSUMMARY\n"
        "1\t4\t0\t0\t0\t0\tGood credit\n"
        "1\t5\t0\t0\t0\t0\tACCOUNT HISTORY\n"
        "1\t6\t0\t0\t0\t0\tAccount 1\n"
        "1\t7\t0\t0\t0\t0\tCOLLECTION CHARGEOFF\n"
        "1\t8\t0\t0\t0\t0\tPUBLIC INFORMATION\n"
        "1\t9\t0\t0\t0\t0\tNone\n"
        "1\t10\t0\t0\t0\t0\tINQUIRIES\n"
        "1\t11\t0\t0\t0\t0\tInq1\n"
        "1\t12\t0\t0\t0\t0\tCREDITOR CONTACTS\n"
        "1\t13\t0\t0\t0\t0\tContact 1\n"
        "1\t14\t0\t0\t0\t0\tSMARTCREDIT\n"
        "1\t15\t0\t0\t0\t0\tAccount # 123\n"
    )
    path.write_text(content, encoding="utf-8")


def test_split_sections(tmp_path: Path) -> None:
    tsv_path = tmp_path / "_debug_full.tsv"
    json_path = tmp_path / "general_info_from_full.json"
    create_full_tsv(tsv_path)

    split_general_info_from_tsv.main(
        ["--full", str(tsv_path), "--json_out", str(json_path)]
    )

    data = json.loads(json_path.read_text())
    sections = data["sections"]
    assert [sec["heading"] for sec in sections] == [
        "PERSONAL INFORMATION",
        "SUMMARY",
        "ACCOUNT HISTORY",
        "PUBLIC INFORMATION",
        "INQUIRIES",
        "CREDITOR CONTACTS",
    ]

    # Account History → Collection Chargeoff should include the ending heading.
    assert [ln["text"] for ln in sections[2]["lines"]] == [
        "ACCOUNT HISTORY",
        "Account 1",
        "COLLECTION CHARGEOFF",
    ]

    # Inquiries should stop before Creditor Contacts.
    assert [ln["text"] for ln in sections[4]["lines"]] == [
        "INQUIRIES",
        "Inq1",
    ]

    # Creditor Contacts should stop before SmartCredit.
    assert [ln["text"] for ln in sections[5]["lines"]] == [
        "CREDITOR CONTACTS",
        "Contact 1",
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


def create_missing_creditor_contacts_tsv(path: Path) -> None:
    content = (
        "page\tline\ty0\ty1\tx0\tx1\ttext\n"
        "1\t1\t0\t0\t0\t0\tPERSONAL INFORMATION\n"
        "1\t2\t0\t0\t0\t0\tName: JOHN DOE\n"
        "1\t3\t0\t0\t0\t0\tSUMMARY\n"
        "1\t4\t0\t0\t0\t0\tGood credit\n"
        "1\t5\t0\t0\t0\t0\tACCOUNT HISTORY\n"
        "1\t6\t0\t0\t0\t0\tAccount 1\n"
        "1\t7\t0\t0\t0\t0\tCOLLECTION CHARGEOFF\n"
        "1\t8\t0\t0\t0\t0\tPUBLIC INFORMATION\n"
        "1\t9\t0\t0\t0\t0\tNone\n"
        "1\t10\t0\t0\t0\t0\tINQUIRIES\n"
        "1\t11\t0\t0\t0\t0\tInq1\n"
        "1\t12\t0\t0\t0\t0\tSMARTCREDIT\n"
        "1\t13\t0\t0\t0\t0\tAccount # 123\n"
    )
    path.write_text(content, encoding="utf-8")


def test_missing_creditor_contacts(tmp_path: Path) -> None:
    tsv_path = tmp_path / "_debug_full.tsv"
    json_path = tmp_path / "general_info_from_full.json"
    create_missing_creditor_contacts_tsv(tsv_path)

    split_general_info_from_tsv.main(
        ["--full", str(tsv_path), "--json_out", str(json_path)]
    )

    data = json.loads(json_path.read_text())
    # Inquiries and Creditor Contacts sections should be skipped.
    assert [sec["heading"] for sec in data["sections"]] == [
        "PERSONAL INFORMATION",
        "SUMMARY",
        "ACCOUNT HISTORY",
        "PUBLIC INFORMATION",
    ]
