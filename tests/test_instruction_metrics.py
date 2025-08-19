from backend.analytics.analytics_tracker import reset_counters, get_counters, set_metric
from backend.core.models import ClientInfo
from tests.helpers.fake_ai_client import FakeAIClient
import backend.core.logic.rendering.instructions_generator as generator
from backend.core.logic.rendering import pdf_renderer


def test_instruction_metrics_emitted(monkeypatch, tmp_path):
    reset_counters()

    def fake_prepare(*a, **k):
        return (
            {
                "client_name": "A",
                "date": "B",
                "accounts_summary": {"problematic": []},
                "per_account_actions": [
                    {"account_ref": "x", "action_sentence": "Pay the balance"}
                ],
            },
            [],
        )

    monkeypatch.setattr(generator, "prepare_instruction_data", fake_prepare)
    monkeypatch.setattr(generator, "save_json_output", lambda *a, **k: None)
    monkeypatch.setattr(generator, "run_compliance_pipeline", lambda *a, **k: None)

    def fake_render(html, path, **kwargs):
        template = kwargs.get("template_name")
        if template:
            set_metric(f"letter.render_ms.{template}", 0)

    monkeypatch.setattr(pdf_renderer, "render_html_to_pdf", fake_render)
    monkeypatch.setenv("LETTERS_ROUTER_PHASED", "1")

    generator.generate_instruction_file(
        ClientInfo.from_dict({"name": "A"}),
        {},
        False,
        tmp_path,
        ai_client=FakeAIClient(),
    )
    counters = get_counters()
    assert counters.get("router.candidate_selected")
    assert counters.get("router.candidate_selected.instruction")
    assert counters.get("router.finalized")
    assert counters.get("router.finalized.instruction")
    assert counters.get("letter_template_selected.instruction_template.html")
    assert counters.get("letter.render_ms.instruction_template.html") is not None
