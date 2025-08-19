from backend.analytics.analytics_tracker import reset_counters, get_counters
from backend.core.models import ClientInfo
from tests.helpers.fake_ai_client import FakeAIClient
import backend.core.logic.rendering.instructions_generator as generator


def test_instruction_metrics_emitted(monkeypatch, tmp_path):
    reset_counters()

    def fake_prepare(*a, **k):
        return (
            {
                "client_name": "A",
                "date": "B",
                "accounts_summary": {},
                "per_account_actions": [{"account_ref": "x", "action_sentence": "do"}],
            },
            [],
        )

    monkeypatch.setattr(generator, "prepare_instruction_data", fake_prepare)
    monkeypatch.setattr(generator, "render_pdf_from_html", lambda *a, **k: None)
    monkeypatch.setattr(generator, "save_json_output", lambda *a, **k: None)
    monkeypatch.setattr(generator, "run_compliance_pipeline", lambda *a, **k: None)

    generator.generate_instruction_file(
        ClientInfo.from_dict({"name": "A"}),
        {},
        False,
        tmp_path,
        ai_client=FakeAIClient(),
    )
    counters = get_counters()
    assert counters.get("letter_template_selected.instruction_template.html")
    assert counters.get("letter.render_ms.instruction_template.html") is not None
