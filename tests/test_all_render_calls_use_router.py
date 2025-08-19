import pytest

from backend.core.logic.rendering.letter_rendering import render_dispute_letter_html
from backend.core.logic.letters.goodwill_rendering import render_goodwill_letter
from backend.core.logic.rendering.instruction_renderer import render_instruction_html
from backend.core.models.letter import LetterContext
from tests.helpers.fake_ai_client import FakeAIClient
import backend.analytics.analytics_tracker as tracker


def test_all_render_calls_use_router(monkeypatch, tmp_path):
    metrics = []

    def fake_counter(name, increment=1):
        metrics.append(name)

    monkeypatch.setattr(
        tracker, "emit_counter", fake_counter
    )
    monkeypatch.setattr(
        "backend.core.logic.rendering.letter_rendering.emit_counter", fake_counter
    )
    monkeypatch.setattr(
        "backend.core.logic.letters.goodwill_rendering.emit_counter", fake_counter
    )
    monkeypatch.setattr(
        "backend.core.logic.rendering.instruction_renderer.emit_counter", fake_counter
    )

    ctx = LetterContext()
    with pytest.raises(ValueError):
        render_dispute_letter_html(ctx, "")

    with pytest.raises(ValueError):
        render_instruction_html({}, template_path="")

    with pytest.raises(ValueError):
        render_goodwill_letter(
            "Creditor",
            {},
            {"legal_name": "A", "session_id": "s", "state": "CA"},
            tmp_path,
            ai_client=FakeAIClient(),
            template_path="",
        )

    assert metrics.count("rendering.missing_template_path") >= 3
