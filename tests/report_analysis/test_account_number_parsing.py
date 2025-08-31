import backend.core.logic.report_analysis.report_parsing as rp


def test_parse_account_block_extracts_account_numbers():
    lines = [
        "Field: TransUnion Experian Equifax",
        "Account #: 517805****** 517805****** 517805******",
    ]
    result = rp.parse_account_block(lines)
    for bureau in ("transunion", "experian", "equifax"):
        bm = result[bureau]
        assert bm["account_number_display"] == "517805******"
        assert bm["account_number_last4"] == "7805"
