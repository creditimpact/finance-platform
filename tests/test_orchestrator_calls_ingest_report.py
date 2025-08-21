import tactical
from backend.core.models import ClientInfo, ProofDocuments
from backend.core import orchestrators
from backend.core.letters import router as letters_router


def test_orchestrator_calls_ingest_report(monkeypatch, tmp_path):
    called = {}

    def fake_ingest_report(account_id, report):
        called['args'] = (account_id, report)
        return []

    monkeypatch.setattr(
        'services.outcome_ingestion.ingest_report.ingest_report',
        fake_ingest_report,
    )

    class DummyConfig:
        ai = {}
        rulebook_fallback_enabled = True
        wkhtmltopdf_path = None
        smtp_server = ""
        smtp_port = 0
        smtp_username = ""
        smtp_password = ""
        export_trace_file = False

    monkeypatch.setattr(orchestrators, 'get_app_config', lambda: DummyConfig())
    monkeypatch.setattr(
        'backend.core.services.ai_client.build_ai_client', lambda *a, **k: None
    )
    monkeypatch.setattr(orchestrators, 'process_client_intake', lambda c, a: ('sess', {}, {}))
    monkeypatch.setattr(orchestrators, 'classify_client_responses', lambda *a, **k: {})
    monkeypatch.setattr(
        orchestrators,
        'normalize_and_tag',
        lambda account_cls, facts, rulebook, account_id=None: facts,
    )
    sections = {"negative_accounts": []}
    bureau_data = {"Experian": {}}
    monkeypatch.setattr(
        orchestrators,
        'analyze_credit_report',
        lambda *a, **k: (tmp_path / 'r.pdf', sections, bureau_data, tmp_path),
    )
    monkeypatch.setattr(orchestrators, 'load_rulebook', lambda: {})
    monkeypatch.setattr(orchestrators, 'generate_strategy_plan', lambda *a, **k: {"accounts": []})
    monkeypatch.setattr(orchestrators, 'plan_next_step', lambda session, tags: tags)
    monkeypatch.setattr(tactical, 'generate_letters', lambda session, tags: None)
    monkeypatch.setattr(orchestrators, 'finalize_outputs', lambda *a, **k: None)
    monkeypatch.setattr(orchestrators, 'save_log_file', lambda *a, **k: None)
    monkeypatch.setattr(orchestrators, 'send_email_with_attachment', lambda *a, **k: None)
    monkeypatch.setattr(orchestrators, 'save_analytics_snapshot', lambda *a, **k: None)
    monkeypatch.setattr(orchestrators, 'update_session', lambda *a, **k: None)
    monkeypatch.setattr(letters_router, 'select_template', lambda *a, **k: type('D', (), {'template_path': ''})())

    client = ClientInfo.from_dict({"name": "Jane", "email": "jane@example.com", "session_id": "sess"})
    report_path = tmp_path / 'report.pdf'
    report_path.write_text('dummy')
    proofs = ProofDocuments.from_dict({"smartcredit_report": str(report_path)})

    orchestrators.run_credit_repair_process(client, proofs, False)

    assert 'args' in called
