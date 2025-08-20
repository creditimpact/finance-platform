import os

from backend.analytics.analytics_tracker import reset_counters, get_counters
from backend.core.models import ClientInfo
from tests.helpers.fake_ai_client import FakeAIClient
import backend.core.logic.rendering.instructions_generator as generator


def test_instruction_validation_missing_actions(monkeypatch, tmp_path):
    reset_counters()

    def fake_prepare(*a, **k):
        return (
            {
                "client_name": "A",
                "date": "B",
                "accounts_summary": {},
                "per_account_actions": [],
            },
            [],
        )

    monkeypatch.setattr(generator, "prepare_instruction_data", fake_prepare)
    monkeypatch.setattr(generator, "render_pdf_from_html", lambda *a, **k: None)
    monkeypatch.setattr(generator, "save_json_output", lambda *a, **k: None)
    monkeypatch.setattr(generator, "run_compliance_pipeline", lambda *a, **k: None)
    monkeypatch.setenv("LETTERS_ROUTER_PHASED", "1")

    generator.generate_instruction_file(
        ClientInfo.from_dict({"name": "A"}),
        {},
        False,
        tmp_path,
        ai_client=FakeAIClient(),
    )
    counters = get_counters()
    # Validation metrics
    assert counters.get("validation.failed")
    assert counters.get("validation.failed.instruction_template.html")
    assert counters.get(
        "validation.failed.instruction_template.html.per_account_actions"
    )
    # Router metrics still emitted
    assert counters.get("router.candidate_selected")
    assert counters.get("router.candidate_selected.instruction")
    assert counters.get(
        "router.candidate_selected.instruction.instruction_template.html"
    )
    assert counters.get("router.finalized")
    assert counters.get("router.finalized.instruction")
    # No render should occur when validation fails
    assert (
        counters.get("letter_template_selected.instruction_template.html")
        is None
    )
    assert counters.get("letter.render_ms.instruction_template.html") is None
