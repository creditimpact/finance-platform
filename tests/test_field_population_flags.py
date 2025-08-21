import importlib
from types import SimpleNamespace

from backend.api.config import AppConfig
from backend.core.models import ClientInfo, ProofDocuments


def reload_orchestrators():
    import backend.api.config as config
    import backend.core.orchestrators as orch

    importlib.reload(config)
    importlib.reload(orch)
    return orch


def test_field_population_disabled(monkeypatch):
    monkeypatch.setenv("ENABLE_FIELD_POPULATION", "0")
    orch = reload_orchestrators()

    calls = []
    metrics = []

    monkeypatch.setattr(
        orch, "apply_field_fillers", lambda *a, **k: calls.append("filled")
    )
    monkeypatch.setattr(orch, "emit_counter", lambda name, tags: metrics.append(name))
    monkeypatch.setattr(
        orch, "process_client_intake", lambda client_info, audit: ("session", {}, {})
    )
    from backend.core.logic.strategy.summary_classifier import (
        RULES_VERSION,
        ClassificationRecord,
    )

    def fake_classify(structured_map, raw_map, client_info, audit, ai_client):
        return {
            "1": ClassificationRecord(
                summary={},
                classification={},
                summary_hash="",
                state=None,
                rules_version=RULES_VERSION,
            )
        }

    monkeypatch.setattr(orch, "classify_client_responses", fake_classify)
    monkeypatch.setattr(orch, "load_rulebook", lambda: {})
    monkeypatch.setattr(
        orch, "analyze_credit_report", lambda *a, **k: (None, {}, {}, None)
    )
    monkeypatch.setattr(
        orch,
        "normalize_and_tag",
        lambda *a, **k: {"action_tag": "dispute", "missing_fields": ["name"]},
    )
    monkeypatch.setattr(orch, "update_session", lambda *a, **k: None)
    monkeypatch.setattr(
        "backend.core.letters.router.select_template",
        lambda tag, ctx, phase=None: SimpleNamespace(
            template_path="tpl", missing_fields=[]
        ),
    )
    monkeypatch.setattr(
        orch, "generate_strategy_plan", lambda *a, **k: {"accounts": []}
    )
    monkeypatch.setattr(orch, "plan_and_generate_letters", lambda *a, **k: None)
    monkeypatch.setattr(orch, "finalize_outputs", lambda *a, **k: None)
    monkeypatch.setattr(orch, "save_log_file", lambda *a, **k: None)
    monkeypatch.setattr(
        "backend.audit.audit.create_audit_logger",
        lambda session_id: SimpleNamespace(
            log_step=lambda *a, **k: None,
            log_error=lambda *a, **k: None,
            log_account=lambda *a, **k: None,
            save=lambda *a, **k: None,
        ),
    )
    monkeypatch.setattr(
        "backend.core.services.ai_client.build_ai_client", lambda *a, **k: None
    )

    app_config = AppConfig(
        ai=None,
        wkhtmltopdf_path="",
        rulebook_fallback_enabled=True,
        export_trace_file=False,
        smtp_server="",
        smtp_port=0,
        smtp_username="",
        smtp_password="",
        celery_broker_url="",
    )

    client = ClientInfo(email="test@example.com")
    proofs = ProofDocuments(smartcredit_report="dummy.pdf")

    orch.run_credit_repair_process(client, proofs, False, app_config=app_config)

    assert calls == []
    assert "finalize.missing_fields_after_population" not in metrics

    monkeypatch.delenv("ENABLE_FIELD_POPULATION", raising=False)
    reload_orchestrators()
