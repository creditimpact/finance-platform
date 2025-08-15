from backend.core.logic.letters.generate_goodwill_letters import (
    generate_goodwill_letter_with_ai,
)
from tests.helpers.fake_ai_client import FakeAIClient


def test_block_goodwill_for_collection(monkeypatch, tmp_path):
    events = []
    monkeypatch.setattr(
        "backend.core.logic.letters.generate_goodwill_letters.emit_event",
        lambda e, p: events.append((e, p)),
    )
    monkeypatch.setattr(
        "backend.core.logic.letters.generate_goodwill_letters.goodwill_prompting.generate_goodwill_letter_draft",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not call GPT")),
    )
    monkeypatch.setattr(
        "backend.core.logic.letters.generate_goodwill_letters.gather_supporting_docs",
        lambda session_id: ("", [], None),
    )
    monkeypatch.setattr(
        "backend.core.logic.rendering.pdf_renderer.render_html_to_pdf",
        lambda html, path: None,
    )
    monkeypatch.setattr(
        "backend.core.logic.compliance.compliance_pipeline.run_compliance_pipeline",
        lambda html, state, session_id, doc_type, ai_client=None: html,
    )
    monkeypatch.setattr(
        "backend.core.logic.letters.generate_goodwill_letters.get_session",
        lambda sid: {},
    )

    accounts = [
        {
            "name": "Collector",
            "account_number": "1",
            "action_tag": "collection",
            "account_id": "1",
        }
    ]
    strategy = {"accounts": accounts}
    client = {"name": "Tester", "session_id": "s1"}

    generate_goodwill_letter_with_ai(
        "Collector",
        accounts,
        client,
        tmp_path,
        ai_client=FakeAIClient(),
        strategy=strategy,
    )

    assert events[-1][1]["policy_override_reason"] == "collection_no_goodwill"
    assert not any(tmp_path.iterdir())
