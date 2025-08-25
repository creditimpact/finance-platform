import backend.core.logic.report_analysis.report_postprocessing as rp
from backend.core.logic.utils.norm import normalize_heading


def test_merge_parser_inquiries_fuzzy_match():
    result = {
        "inquiries": [
            {"creditor_name": "Capital One", "date": "01/2024", "bureau": "Experian"}
        ]
    }
    parsed = [
        {"creditor_name": "CAPTL ONE", "date": "01/2024", "bureau": "Experian"}
    ]
    raw_map = {normalize_heading(p["creditor_name"]): p["creditor_name"] for p in parsed}
    rp._merge_parser_inquiries(result, parsed, raw_map)
    assert result["inquiries"] == [
        {"creditor_name": "CAPTL ONE", "date": "01/2024", "bureau": "Experian"}
    ]
