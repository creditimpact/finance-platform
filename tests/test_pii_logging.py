from pathlib import Path

from backend.audit.audit import AuditLevel, create_audit_logger
from backend.analytics.analytics_tracker import save_analytics_snapshot


def test_audit_save_masks_pii(tmp_path: Path):
    audit = create_audit_logger("sess1", level=AuditLevel.VERBOSE)
    audit.log_step(
        "strategist_invocation",
        {
            "ssn": "123-45-6789",
            "ssn_last4": "ssn 9876",
            "acct": "0000-1111-2222-3333",
            "acct_spaced": "0000 1111 2222 3333",
        },
    )
    path = audit.save(tmp_path)
    data = path.read_text()
    assert "123-45-6789" not in data
    assert "ssn 9876" not in data
    assert "0000-1111-2222-3333" not in data
    assert "0000 1111 2222 3333" not in data
    assert "***-**-6789" in data
    assert "ssn ***-**-9876" in data
    assert "****3333" in data


def test_snapshot_masks_pii(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    client_info = {"name": "Test 123-45-6789 ssn 9876"}
    report_summary = {"strategic_recommendations": ["Call 0000 1111 2222 3333"]}
    save_analytics_snapshot(client_info, report_summary)
    file = next(Path("analytics_data").glob("*.json"))
    text = file.read_text()
    assert "123-45-6789" not in text
    assert "ssn 9876" not in text
    assert "0000 1111 2222 3333" not in text
    assert "***-**-6789" in text
    assert "ssn ***-**-9876" in text
    assert "****3333" in text
