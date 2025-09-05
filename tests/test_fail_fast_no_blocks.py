from pathlib import Path

import pytest

from backend.core import orchestrators as orch
from backend.core.case_store.errors import CaseStoreError


def test_analyze_credit_report_fails_when_no_blocks(monkeypatch, tmp_path):
    sample_pdf = Path("tests/fixtures/sample_block.pdf")
    session_id = "sess-no-blocks"

    monkeypatch.setattr(
        "backend.api.session_manager.update_session", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "backend.core.logic.compliance.upload_validator.move_uploaded_file",
        lambda path, sid: sample_pdf,
    )
    monkeypatch.setattr(
        "backend.core.logic.compliance.upload_validator.is_safe_pdf", lambda p: True
    )
    monkeypatch.setattr(orch, "extract_and_cache_text", lambda *a, **k: None)

    def fake_export(sid, pdf_path):
        out_dir = Path("traces") / "blocks" / sid
        if out_dir.exists():
            import shutil

            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        return []

    monkeypatch.setattr(orch, "export_account_blocks", fake_export)

    audit = type("Audit", (), {"level": orch.AuditLevel.ESSENTIAL})()
    log_messages: list[str] = []

    with pytest.raises(CaseStoreError) as exc:
        orch.analyze_credit_report(
            {"smartcredit_report": sample_pdf},
            session_id,
            {},
            audit,
            log_messages,
            ai_client=None,
        )

    assert exc.value.args[0] == "no_blocks"
