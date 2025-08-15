from pathlib import Path

from backend.core.logic.letters.generate_custom_letters import generate_custom_letter
from backend.core.logic.strategy.summary_classifier import ClassificationRecord
from backend.analytics.analytics_tracker import get_counters, reset_counters
from tests.helpers.fake_ai_client import FakeAIClient


def test_custom_prompt_policy_conflict(monkeypatch, tmp_path: Path):
    events = []
    monkeypatch.setattr(
        "backend.core.logic.letters.generate_custom_letters.emit_event",
        lambda e, p: events.append((e, p)),
    )
    monkeypatch.setattr(
        "backend.core.logic.letters.generate_custom_letters.gather_supporting_docs",
        lambda session_id: ("", [], None),
    )
    monkeypatch.setattr(
        "backend.core.logic.letters.generate_custom_letters.pdfkit.from_string",
        lambda html, path, configuration=None, options=None: None,
    )
    monkeypatch.setattr(
        "backend.core.logic.letters.generate_custom_letters.get_session",
        lambda sid: {"structured_summaries": {"1": {"account_id": "1", "debt_type": "credit"}}},
    )

    fake = FakeAIClient()
    fake.add_chat_response(
        "Please accept this goodwill payment and delete the debt."
    )
    fake.add_chat_response("This is a dispute letter without policy issues.")

    account = {
        "name": "Bank",
        "account_number": "1",
        "action_tag": "dispute",
        "account_id": "1",
        "forbidden_actions": ["Goodwill"],
    }
    client = {"legal_name": "Tester", "session_id": "s1"}

    classification_map = {
        "1": ClassificationRecord(
            summary={}, classification={"category": "error", "action_tag": "dispute"}, summary_hash=""
        )
    }

    reset_counters()
    generate_custom_letter(
        account,
        client,
        tmp_path,
        audit=None,
        ai_client=fake,
        classification_map=classification_map,
    )

    text = (tmp_path / "Bank_custom_gpt_response.txt").read_text()
    assert "goodwill" not in text.lower()
    assert events[-1][1]["override_reason"] == "custom_prompt_policy_conflict"
    assert events[-1][1]["action_tag_after"] == "dispute"
    counters = get_counters()
    assert counters["policy_override_reason.custom_prompt_policy_conflict"] == 1
