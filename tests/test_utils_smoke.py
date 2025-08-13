from pathlib import Path

from backend.core.logic.utils.file_paths import safe_filename
from backend.core.logic.utils.inquiries import extract_inquiries
from backend.core.logic.utils.names_normalization import normalize_bureau_name
from backend.core.logic.utils.note_handling import get_client_address_lines
from backend.core.logic.utils.pdf_ops import gather_supporting_docs_text
from backend.core.logic.utils.pii import redact_pii
from backend.core.logic.utils.report_sections import filter_sections_by_bureau
from backend.core.logic.utils.text_parsing import has_late_indicator


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

    red = redact_pii("email test@example.com phone 555-111-2222 ssn 123-45-6789")
    assert "test@example.com" not in red
    assert "555-111-2222" not in red
    assert "123-45-6789" not in red
