from backend.core.logic.report_analysis.block_segmenter import segment_account_blocks


def test_no_blocks_for_non_account_headings():
    txt = (
        "\n"
        "BANKAMERICA\n"
        "Account # ****1234\n"
        "Date Opened: 01/2020\n"
        "Payment Status: Current\n"
        "TransUnion Experian Equifax\n"
        "\n"
        "Individual\n"
        "Account not disputed\n"
        "Bank - Mortgage Loans\n"
        "\n"
        "JPMCB CARD\n"
        "Account # ****5678\n"
        "Date Opened: 02/2021\n"
    )
    blocks = segment_account_blocks(txt)
    heads = [
        b["heading"].upper().strip()
        for b in blocks
        if b.get("meta", {}).get("block_type") == "account"
    ]
    assert "BANKAMERICA" in heads
    assert "JPMCB CARD" in heads
    assert "INDIVIDUAL" not in heads
    assert "ACCOUNT NOT DISPUTED" not in heads
    assert "BANK - MORTGAGE LOANS" not in heads

