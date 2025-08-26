import json

from backend.core.logic.report_analysis.candidate_logger import CandidateTokenLogger


def test_candidate_tokens_written(tmp_path):
    logger = CandidateTokenLogger()
    acc = {
        "account_status": "Open",
        "payment_status": "Current",
        "account_description": "Credit Card",
        "creditor_remarks": "Test remark",
    }
    logger.collect(acc)
    logger.save(tmp_path)
    path = tmp_path / "candidate_tokens.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["account_status"] == ["Open"]
    assert data["creditor_remarks"] == ["Test remark"]
