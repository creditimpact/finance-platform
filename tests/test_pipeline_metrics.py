from pathlib import Path
import types

from backend.analytics.analytics_tracker import reset_counters, get_counters
from backend.core.logic.guardrails import (
    generate_letter_with_guardrails,
    fix_draft_with_guardrails,
)
from backend.core.logic.rendering import pdf_renderer
from tests.helpers.fake_ai_client import FakeAIClient


def test_pipeline_metrics(monkeypatch, tmp_path):
    reset_counters()

    def fake_config(**kwargs):
        return types.SimpleNamespace()

    def fake_from_string(html, path, configuration=None, options=None):
        Path(path).write_bytes(b"PDF")

    monkeypatch.setattr(pdf_renderer.pdfkit, "configuration", fake_config)
    monkeypatch.setattr(pdf_renderer.pdfkit, "from_string", fake_from_string)

    ai = FakeAIClient()
    text, _viol, _ = generate_letter_with_guardrails(
        "hello", None, {"debt_type": "cc", "dispute_reason": "err"}, "", "custom", ai_client=ai
    )
    text += " 12345 "
    fix_draft_with_guardrails(text, None, {}, "", "custom", ai_client=ai)

    out = tmp_path / "out.pdf"
    pdf_renderer.render_html_to_pdf(
        "<p>hi</p>", str(out), template_name="test_template.html"
    )

    counters = get_counters()
    assert counters.get("ai.tokens.candidate") is not None
    assert counters.get("ai.tokens.finalize") is not None
    assert counters.get("ai.tokens.render") is not None
    assert counters.get("ai.cost.candidate") is not None
    assert counters.get("ai.cost.finalize") is not None
    assert counters.get("ai.cost.render") is not None
    assert (
        counters.get("letter.render_ms.test_template.html") is not None
        and counters["letter.render_ms.test_template.html"] >= 0
    )
