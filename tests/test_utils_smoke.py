from pathlib import Path

from logic.utils import (
    names_normalization,
    note_handling,
    file_paths,
    text_parsing,
    inquiries,
    pdf_ops,
    report_sections,
)


def test_utils_smoke(tmp_path: Path):
    assert names_normalization.normalize_bureau_name("tu") == "TransUnion"
    assert file_paths.safe_filename("a:b") == "a_b"

    notes, _ = note_handling.analyze_custom_notes({"Cap One": "note"}, ["Cap One"])
    norm_name = names_normalization.normalize_creditor_name("Cap One")
    assert notes[norm_name] == "note"
    assert note_handling.get_client_address_lines({"address": "123 A St, Town"})

    assert text_parsing.has_late_indicator({"status": "30 days late"})

    text = (
        "Inquiries\nCreditor Name Date of Inquiry Credit Bureau\nCap One 01/01/23 Experian"
    )
    assert inquiries.extract_inquiries(text)

    # PDF helpers should handle missing files gracefully
    assert pdf_ops.gather_supporting_docs_text("") == ""

    sections = {"negative_accounts": [{"name": "Cap One", "bureaus": ["Experian"]}]}
    filtered = report_sections.filter_sections_by_bureau(sections, "Experian")
    assert filtered["disputes"]
