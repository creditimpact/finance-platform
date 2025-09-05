import pytest
from dataclasses import replace
from pathlib import Path


def test_orchestrator_fails_when_no_cases(monkeypatch):
    import backend.core.config.flags as flags
    from backend.core.orchestrators import _StubAIClient

    # Enforce CASE_FIRST requirement
    monkeypatch.setattr(
        flags,
        "FLAGS",
        replace(flags.FLAGS, case_first_build_required=True),
    )

    # Bypass file system and AI dependencies
    monkeypatch.setattr(
        "backend.core.logic.compliance.upload_validator.move_uploaded_file",
        lambda path, session_id: Path(path),
    )
    monkeypatch.setattr(
        "backend.core.logic.compliance.upload_validator.is_safe_pdf",
        lambda path: True,
    )
    monkeypatch.setattr(
        "backend.core.logic.report_analysis.analyze_report.analyze_credit_report",
        lambda *a, **k: {},
    )
    monkeypatch.setattr(
        "backend.core.logic.report_analysis.extractors.accounts.build_account_cases",
        lambda sid: None,
    )
    monkeypatch.setattr("backend.core.orchestrators.list_accounts", lambda sid: [])
    monkeypatch.setattr(
        "backend.core.orchestrators.get_ai_client", lambda: _StubAIClient()
    )
    monkeypatch.setattr(
        "backend.core.logic.report_analysis.text_provider._extract_text_per_page",
        lambda p: ["stub"],
    )
    monkeypatch.setattr(
        "backend.core.orchestrators.write_text_trace", lambda *a, **k: ""
    )
    monkeypatch.setattr(
        "backend.api.session_manager.update_session", lambda sid, **kw: None
    )

    from backend.core.orchestrators import extract_problematic_accounts_from_report
    from backend.core.case_store.errors import CaseStoreError

    with pytest.raises(CaseStoreError) as e:
        extract_problematic_accounts_from_report("dummy.pdf", "s-no-cases")
    assert "case_build_failed" in str(e.value)
