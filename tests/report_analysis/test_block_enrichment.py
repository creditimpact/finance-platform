from backend.core.logic.report_analysis.block_exporter import enrich_block


def test_enrich_block_extracts_fields():
    lines = [
        "AMERICAN EXPRESS",
        "Transunion Experian Equifax",
        "Account # 1234****** 1234****** 1234******",
        "High Balance: $261 $261 $261",
        "Payment Status: Current Current Current",
        "Credit Limit: $1,000 $1,000 $1,000",
    ]
    blk = {"heading": "AMEX", "lines": lines}
    res = enrich_block(blk)
    assert res["fields"]["transunion"]["payment_status"] == "Current"
    assert "****" in res["fields"]["experian"]["account_number_display"]
    assert res["fields"]["equifax"]["credit_limit"] == "$1,000"


def test_enrich_block_original_creditor_variants():
    lines = [
        "AMERICAN EXPRESS",
        "Transunion Experian Equifax",
        "Original Creditor 02: PALISADES FUNDING CORP  XP CREDITOR  EQ CREDITOR",
    ]
    blk = {"heading": "AMEX", "lines": lines}
    res = enrich_block(blk)
    fields = res["fields"]

    assert fields["transunion"]["original_creditor"] == "PALISADES FUNDING CORP"
    assert fields["experian"]["original_creditor"] == "XP CREDITOR"
    assert fields["equifax"]["original_creditor"] == "EQ CREDITOR"
