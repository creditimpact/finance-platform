import json
from pathlib import Path

from scripts import split_general_info_from_tsv


def write_tsv(path: Path, rows: list[tuple[int, int, int, str]]) -> None:
    """Helper to write a minimal TSV file for testing.

    ``rows`` is an iterable of ``(page, line, x0, text)`` tuples.
    Other TSV columns are filled with dummy ``0`` values.
    """

    header = "page\tline\ty0\ty1\tx0\tx1\ttext\n"
    lines = [
        f"{p}\t{l}\t0\t0\t{x}\t0\t{t}\n" for (p, l, x, t) in rows
    ]
    path.write_text(header + "".join(lines), encoding="utf-8")


def test_split_sections_tokenised_headings(tmp_path: Path) -> None:
    tsv_path = tmp_path / "full.tsv"
    json_path = tmp_path / "out.json"

    rows = [
        (1, 1, 0, "Personal"),
        (1, 1, 10, "Information"),
        (1, 2, 0, "Name: JOHN DOE"),
        (1, 3, 0, "SUMMARY"),
        (1, 4, 0, "Good credit"),
        (1, 5, 0, "Public Records:"),  # should not trigger Public Information
        (1, 6, 0, "ACCOUNT"),
        (1, 6, 10, "HISTORY"),
        (1, 7, 0, "Account 1"),
        (1, 8, 0, "Collection"),
        (1, 8, 10, "Chargeof"),  # tolerate truncated heading
        (1, 9, 0, "Debt 1"),
        (1, 10, 0, "PUBLIC"),
        (1, 10, 10, "INFORMATION"),
        (1, 11, 0, "None"),
        (1, 12, 0, "INQUIRIES"),
        (1, 13, 0, "Inq1"),
        (1, 14, 0, "CREDITOR"),
        (1, 14, 10, "CONTACTS"),
        (1, 15, 0, "Contact 1"),
        (1, 16, 0, "smartcredit.com"),  # footer terminator
        (1, 17, 0, "Account # 123"),
    ]

    write_tsv(tsv_path, rows)

    split_general_info_from_tsv.main(
        ["--full", str(tsv_path), "--json_out", str(json_path)]
    )

    data = json.loads(json_path.read_text())
    sections = data["sections"]

    assert [s["heading"] for s in sections] == [
        "Personal Information",
        "Summary",
        "Account History",
        "Collection / Chargeoff",
        "Public Information",
        "Inquiries",
        "Creditor Contacts",
    ]

    # Account History should include the collections title line
    account_lines = [ln["text"] for ln in sections[2]["lines"]]
    assert account_lines[-1].startswith("Collection")

    collection_lines = [ln["text"] for ln in sections[3]["lines"]]
    assert collection_lines == ["Debt 1"]


def test_missing_collections_and_footer(tmp_path: Path) -> None:
    tsv_path = tmp_path / "full.tsv"
    json_path = tmp_path / "out.json"

    rows = [
        (1, 1, 0, "PERSONAL INFORMATION"),
        (1, 2, 0, "SUMMARY"),
        (1, 3, 0, "ACCOUNT HISTORY"),
        (1, 4, 0, "Account 1"),
        (1, 5, 0, "PUBLIC INFORMATION"),
        (1, 6, 0, "INQUIRIES"),
        (1, 7, 0, "CREDITOR CONTACTS"),
        (1, 8, 0, "Service Agreement"),  # footer via synonym
    ]

    write_tsv(tsv_path, rows)

    split_general_info_from_tsv.main(
        ["--full", str(tsv_path), "--json_out", str(json_path)]
    )

    data = json.loads(json_path.read_text())
    # Collections section is missing; others present until footer
    assert [s["heading"] for s in data["sections"]] == [
        "Personal Information",
        "Summary",
        "Account History",
        "Public Information",
        "Inquiries",
        "Creditor Contacts",
    ]


def test_no_sections(tmp_path: Path) -> None:
    tsv_path = tmp_path / "full.tsv"
    json_path = tmp_path / "out.json"

    rows = [
        (1, 1, 0, "Name: JOHN DOE"),
        (1, 2, 0, "Account # 123"),
    ]

    write_tsv(tsv_path, rows)

    split_general_info_from_tsv.main(
        ["--full", str(tsv_path), "--json_out", str(json_path)]
    )

    data = json.loads(json_path.read_text())
    assert data["sections"] == []


def test_general_info_ignores_summary_occurrences_and_splits_late_headings(
    tmp_path: Path,
) -> None:
    tsv_path = tmp_path / "full.tsv"
    json_path = tmp_path / "out.json"

    rows = [
        (1, 1, 0, "PERSONAL INFORMATION"),
        (1, 2, 0, "Name: JOHN DOE"),
        (1, 3, 0, "SUMMARY"),
        (1, 4, 0, "Public Information"),  # inside Summary
        (1, 5, 0, "Inquiries"),  # inside Summary
        (1, 6, 0, "ACCOUNT HISTORY"),
        (1, 7, 0, "Account 1"),
        (1, 8, 0, "PUBLIC INFORMATION"),
        (1, 9, 0, "Detail"),
        (1, 10, 0, "INQUIRIES"),
        (1, 11, 0, "Detail"),
        (1, 12, 0, "smartcredit.com"),
    ]

    write_tsv(tsv_path, rows)

    split_general_info_from_tsv.main(
        ["--full", str(tsv_path), "--json_out", str(json_path)]
    )

    data = json.loads(json_path.read_text())
    sections = data["sections"]

    assert [s["heading"] for s in sections] == [
        "Personal Information",
        "Summary",
        "Account History",
        "Public Information",
        "Inquiries",
    ]

    assert sections[3]["line_start"] == 8
    assert sections[4]["line_start"] == 10
    assert data["summary_filter_applied"] is True


def test_general_info_no_filter_when_anchors_missing(tmp_path: Path) -> None:
    tsv_path = tmp_path / "full.tsv"
    json_path = tmp_path / "out.json"

    rows = [
        (1, 1, 0, "PERSONAL INFORMATION"),
        (1, 2, 0, "Name: JOHN DOE"),
        (1, 3, 0, "SUMMARY"),
        (1, 4, 0, "Public Information"),  # treated as heading
        (1, 5, 0, "Inquiries"),  # treated as heading
        (1, 6, 0, "smartcredit.com"),
    ]

    write_tsv(tsv_path, rows)

    split_general_info_from_tsv.main(
        ["--full", str(tsv_path), "--json_out", str(json_path)]
    )

    data = json.loads(json_path.read_text())
    sections = data["sections"]

    assert [s["heading"] for s in sections] == [
        "Personal Information",
        "Summary",
        "Public Information",
        "Inquiries",
    ]
    assert data["summary_filter_applied"] is False

