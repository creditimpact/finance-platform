import backend.core.logic.report_analysis.report_postprocessing as rp


def test_merge_parser_inquiries_keeps_known_creditor_name():
    result = {
        "inquiries": [
            {"creditor_name": "Local Credit Union", "date": "03/2024", "bureau": "Equifax"}
        ]
    }
    parsed = []

    rp._merge_parser_inquiries(result, parsed)

    assert result["inquiries"][0]["creditor_name"] == "Local Credit Union"
