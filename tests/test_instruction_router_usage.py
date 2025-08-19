from pathlib import Path
import pytest

from backend.core.models import ClientInfo
from tests.helpers.fake_ai_client import FakeAIClient
import backend.core.logic.rendering.instructions_generator as generator
from backend.core.letters.router import TemplateDecision


def test_instruction_requires_router(monkeypatch, tmp_path):
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

    def fake_select(tag, ctx, phase):
        return TemplateDecision(template_path=None, required_fields=[], missing_fields=[], router_mode="auto_route")

    called = {"built": False}

    def fake_build(ctx, template_path):
        called["built"] = True
        return "html"

    monkeypatch.setattr(generator, "prepare_instruction_data", fake_prepare)
    monkeypatch.setattr(generator, "select_template", fake_select)
    monkeypatch.setattr(generator, "build_instruction_html", fake_build)
    monkeypatch.setattr(generator, "render_pdf_from_html", lambda *a, **k: None)
    monkeypatch.setattr(generator, "save_json_output", lambda *a, **k: None)
    monkeypatch.setattr(generator, "run_compliance_pipeline", lambda *a, **k: None)

    with pytest.raises(ValueError):
        generator.generate_instruction_file(
            ClientInfo.from_dict({"name": "A"}),
            {},
            False,
            tmp_path,
            ai_client=FakeAIClient(),
        )
    assert not called["built"]
