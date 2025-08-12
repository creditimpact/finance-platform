from pathlib import Path

from logic.utils.names_normalization import (
    normalize_bureau_name,
    normalize_creditor_name,
)
from logic.utils.note_handling import get_client_address_lines
from logic.utils.file_paths import safe_filename
from logic.utils.text_parsing import has_late_indicator
from logic.utils.inquiries import extract_inquiries
from logic.utils.pdf_ops import gather_supporting_docs_text
from logic.utils.report_sections import filter_sections_by_bureau


def test_utils_smoke(tmp_path: Path):
    assert normalize_bureau_name("tu") == "TransUnion"
    assert safe_filename("a:b") == "a_b"

    assert get_client_address_lines({"address": "123 A St, Town"})

    assert has_late_indicator({"status": "30 days late"})

    text = "Inquiries\nCreditor Name Date of Inquiry Credit Bureau\nCap One 01/01/23 Experian"
    assert extract_inquiries(text)

    # PDF helpers should handle missing files gracefully
    assert gather_supporting_docs_text("") == ""

    sections = {"negative_accounts": [{"name": "Cap One", "bureaus": ["Experian"]}]}
    filtered = filter_sections_by_bureau(sections, "Experian")
    assert filtered["disputes"]
