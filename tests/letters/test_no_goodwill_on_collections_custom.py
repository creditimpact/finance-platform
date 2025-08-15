from pathlib import Path

import pytest

from backend.core.logic.letters.generate_custom_letters import generate_custom_letter
from backend.core.logic.strategy.summary_classifier import ClassificationRecord
from tests.helpers.fake_ai_client import FakeAIClient


def test_block_goodwill_on_collection(monkeypatch, tmp_path: Path):
    events = []
    monkeypatch.setattr(
        "backend.core.logic.letters.generate_custom_letters.emit_event",
        lambda e, p: events.append((e, p)),
    )
    monkeypatch.setattr(
        "backend.core.logic.letters.generate_custom_letters.call_gpt_for_custom_letter",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not call GPT")),
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
        lambda sid: {},
    )

    account = {
        "name": "Collector",
        "account_number": "1",
        "action_tag": "goodwill",
        "account_id": "1",
    }
    classification_map = {
        "1": ClassificationRecord(
            summary={}, classification={"category": "collection", "action_tag": "goodwill"}, summary_hash=""
        )
    }
    client = {"name": "Tester", "session_id": "s1"}

    fake = FakeAIClient()
    generate_custom_letter(
        account,
        client,
        tmp_path,
        audit=None,
        ai_client=fake,
        classification_map=classification_map,
    )

    assert events[-1][1]["policy_override_reason"] == "collection_no_goodwill"
    assert not any(tmp_path.iterdir())
