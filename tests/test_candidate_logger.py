import json

from backend.core.logic.report_analysis.candidate_logger import CandidateTokenLogger


def test_candidate_tokens_written(tmp_path):
    logger = CandidateTokenLogger()
    acc = {
        "balance_owed": 100.5,
        "account_rating": "Late 30",
        "account_description": "Credit Card 1234",
        "dispute_status": "None",
        "creditor_type": "Bank",
        "account_status": "Open",
        "payment_status": "Current",
        "creditor_remarks": "Remark 5678",
        "account_type": "Credit",
        "credit_limit": 5000,
        "late_payments": {"Equifax": {"30": 1}},
        "past_due_amount": 200,
    }
    logger.collect(acc)
    logger.save(tmp_path)
    path = tmp_path / "candidate_tokens.json"
    assert path.exists()
    data = json.loads(path.read_text())
    expected_fields = {
        "balance_owed",
        "account_rating",
        "account_description",
        "dispute_status",
        "creditor_type",
        "account_status",
        "payment_status",
        "creditor_remarks",
        "account_type",
        "credit_limit",
        "late_payments",
        "past_due_amount",
    }
    assert expected_fields == set(data.keys())
    assert data["account_description"] == ["Credit Card XXXX"]
    assert "Equifax:30:1" in data["late_payments"]
