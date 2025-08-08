from logic.goodwill_preparation import (
    select_goodwill_candidates,
    prepare_account_summaries,
)


def test_select_goodwill_candidates_detects_late_accounts():
    bureau_data = {
        "exp": {
            "goodwill": [
                {"name": "Bank", "status": "Open", "late_payments": {"EXP": {"30": 1}}}
            ]
        }
    }
    client_info = {}
    result = select_goodwill_candidates(client_info, bureau_data)
    assert "Bank" in result and len(result["Bank"]) == 1


def test_prepare_account_summaries_merges_and_enriches():
    accounts = [
        {
            "name": "Chase",
            "account_number": "1234",
            "status": "Open",
            "late_payments": {"EXP": {"30": 1}},
            "account_id": "a1",
        },
        {"name": "Chase", "acct_number": "1234", "status": "Open"},
    ]
    structured = {"a1": {"account_id": "a1", "dispute_type": "goodwill"}}
    summaries = prepare_account_summaries(accounts, structured, state="CA")
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["account_number"] == "1234"
    assert "late_history" in summary
    assert summary["dispute_reason"] == "goodwill"
    assert summary["state_hook"] == "California Consumer Credit Reporting Agencies Act"
