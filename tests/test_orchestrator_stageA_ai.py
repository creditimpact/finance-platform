import json
from importlib import reload

import backend.core.orchestrators as orch


def test_problem_accounts_only(monkeypatch, caplog):
    sections = {
        "problem_accounts": [
            {
                "normalized_name": "acc1",
                "primary_issue": "collection",
                "issue_types": ["collection"],
                "problem_reasons": ["status"],
                "confidence": 0.9,
                "tier": 1,
                "decision_source": "ai",
            },
            {
                "normalized_name": "acc2",
                "primary_issue": "unknown",
                "issue_types": [],
                "problem_reasons": ["utilization"],
                "confidence": 0.0,
                "tier": 0,
                "decision_source": "rules",
            },
        ],
        "all_accounts": [
            {"normalized_name": "acc1"},
            {"normalized_name": "acc2"},
            {"normalized_name": "acc3"},
        ],
    }
    monkeypatch.setattr(
        "backend.core.logic.compliance.upload_validator.move_uploaded_file",
        lambda path, session_id: path,
    )
    monkeypatch.setattr(
        "backend.core.logic.compliance.upload_validator.is_safe_pdf", lambda path: True
    )
    monkeypatch.setattr(
        "backend.core.orchestrators.update_session", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "backend.core.logic.report_analysis.analyze_report.analyze_credit_report",
        lambda *a, **k: sections,
    )
    monkeypatch.setenv("PROBLEM_DETECTION_ONLY", "1")
    reload(orch)
    caplog.set_level("INFO")
    result = orch.extract_problematic_accounts_from_report("dummy.pdf")
    assert result["problem_accounts"] == sections["problem_accounts"]
    logs = [r.message for r in caplog.records if "stageA_problem_decision" in r.message]
    assert len(logs) == 2
    for entry in logs:
        data = json.loads(entry.split(" ", 1)[1])
        assert "decision_source" in data
