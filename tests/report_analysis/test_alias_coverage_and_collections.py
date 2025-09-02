from backend.core.logic.report_analysis.report_parsing import (
    _assign_std,
    parse_collection_block,
)


def test_alias_variants_map_to_std_fields():
    aliases = {
        "creditor category": "creditor_type",
        "dispute flag": "dispute_status",
        "loan term": "term_length",
        "2-year payment history": "two_year_payment_history",
        "7-year days late": "seven_year_days_late",
    }
    for alias, std in aliases.items():
        dst = {}
        _assign_std(dst, alias, "x")
        assert std in dst


def test_collection_lines_without_colon_are_parsed():
    lines = [
        "PALISADES FU",
        "Transunion Experian Equifax",
        "Account # M20191************ M20191************ M20191************",
        "High Balance $23,025 $23,025 $23,025",
        "Date Opened 01/23/2018 01/23/2018 01/23/2018",
        "Past Due Amount $0 $0 $0",
        "Account Status Collection Collection Collection",
    ]
    res = parse_collection_block(lines)
    tu = res["transunion"]
    filled = sum(1 for v in tu.values() if v is not None)
    assert filled >= 5
    assert tu["high_balance"]["normalized"] == 23025.0
    assert tu["date_opened"]["normalized"] == "2018-01-23"


def test_collection_order_carry_forward_without_header():
    lines = [
        "Account # 1 2 3",
        "High Balance $100 $200 $300",
    ]
    order = ["experian", "equifax", "transunion"]
    res = parse_collection_block(lines, bureau_order=order)
    assert res["experian"]["high_balance"]["normalized"] == 100.0
    assert res["equifax"]["high_balance"]["normalized"] == 200.0
    assert res["transunion"]["high_balance"]["normalized"] == 300.0


def test_currency_and_dates_normalized_in_collections():
    lines = [
        "Transunion Experian Equifax",
        "High Balance $23,025 $23,025 $23,025",
        "Date Opened 1/1/2018 1/1/2018 1/1/2018",
    ]
    res = parse_collection_block(lines)
    assert res["transunion"]["high_balance"]["normalized"] == 23025.0
    assert res["transunion"]["date_opened"]["normalized"] == "2018-01-01"
