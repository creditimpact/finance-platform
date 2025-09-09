from backend.core.logic.report_analysis.block_segmenter import segment_account_blocks


def _sample_text():
    return (
        "TOTAL ACCOUNTS\n"
        "Some summary line here\n"
        "BANKAMERICA\n"
        "TransUnion Experian Equifax\n"
        "Account # ****1234 ****5678 ****9012\n"
        "Payment Status: Current Current Delinquent\n"
        "CLOSED OR PAID ACCOUNT/ZERO\n"
        "Random summary details\n"
        "AMEX\n"
        "Account # ****0000\n"
        "Date Opened: 01/01/2020\n"
    )


def test_segments_accounts_vs_summaries():
    blocks = segment_account_blocks(_sample_text())
    assert blocks
    types = [b.get("meta", {}).get("block_type") for b in blocks]
    assert "summary" in types
    assert "account" in types
    # Ensure known summaries are marked as such
    first = blocks[0]
    assert first["heading"].strip().upper() == "TOTAL ACCOUNTS"
    assert first["meta"]["block_type"] == "summary"


def test_account_block_contains_fields_triggers():
    blocks = segment_account_blocks(_sample_text())
    acc_blocks = [b for b in blocks if b.get("meta", {}).get("block_type") == "account"]
    assert acc_blocks
    for b in acc_blocks:
        probe = " ".join(b["lines"]).lower()
        assert (
            ("transunion" in probe and "experian" in probe and "equifax" in probe)
            or "account #" in probe
            or "date opened" in probe
            or "payment status" in probe
            or "balance owed" in probe
        )


def test_index_shape_unchanged(monkeypatch, tmp_path):
    # This test is basic because exporter formatting is verified elsewhere
    # Here we ensure our segmentation returns plain blocks without changing meta keys unexpectedly
    blocks = segment_account_blocks(_sample_text())
    for b in blocks:
        assert set(b.keys()) >= {"heading", "lines", "meta"}

