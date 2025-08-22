import backend.core.logic.report_analysis.report_postprocessing as rp
from backend.core.logic.utils.names_normalization import normalize_creditor_name


def test_merge_parser_inquiries_preserves_creditor_name():
    result = {
        "inquiries": [
            {"creditor_name": "cap one", "date": "01/2024", "bureau": "Experian"}
        ]
    }
    parsed = [
        {"creditor_name": "Cap One", "date": "01/2024", "bureau": "Experian"},
        {"creditor_name": "Chase Bank", "date": "02/2024", "bureau": "TransUnion"},
    ]
    raw_map = {
        normalize_creditor_name(p["creditor_name"]): p["creditor_name"] for p in parsed
    }
    rp._merge_parser_inquiries(result, parsed, raw_map)

    names = [inq["creditor_name"] for inq in result["inquiries"]]
    assert "Cap One" in names
    assert "Chase Bank" in names
