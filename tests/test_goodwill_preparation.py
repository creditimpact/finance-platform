from backend.core.logic.letters.goodwill_preparation import (
    prepare_account_summaries, select_goodwill_candidates)
from tests.helpers.fake_ai_client import FakeAIClient


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
    from backend.core.logic.strategy.summary_classifier import (
        ClassificationRecord,
        summary_hash,
    )
    record = ClassificationRecord(
        structured["a1"],
        {
            "category": "goodwill",
            "legal_tag": "FCRA §623(a)(1)",
            "dispute_approach": "goodwill_adjustment",
            "tone": "conciliatory",
            "state_hook": "California Consumer Credit Reporting Agencies Act",
        },
        summary_hash(structured["a1"]),
    )
    summaries = prepare_account_summaries(
        accounts,
        structured,
        {"a1": record},
        state="CA",
        session_id="sess",
    )
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["account_number"] == "1234"
    assert "late_history" in summary
    assert summary["dispute_reason"] == "goodwill"
    assert summary["state_hook"] == "California Consumer Credit Reporting Agencies Act"
