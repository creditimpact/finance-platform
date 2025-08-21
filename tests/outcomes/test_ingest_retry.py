from __future__ import annotations

import services.outcome_ingestion as ingestion
from backend.outcomes.models import OutcomeEvent, Outcome
from types import SimpleNamespace


def test_dead_letter_queue(monkeypatch):
    calls = {"n": 0}

    def boom(*args, **kwargs):
        calls["n"] += 1
        raise RuntimeError("fail")

    monkeypatch.setattr(ingestion, "planner", SimpleNamespace(handle_outcome=boom))

    ing_event = OutcomeEvent(
        outcome_id="1",
        account_id="a1",
        cycle_id=0,
        family_id="f1",
        outcome=Outcome.VERIFIED,
    )

    ingestion.DEAD_LETTER_QUEUE.clear()
    ingestion.ingest({"session_id": "s1"}, ing_event, max_retries=3)

    assert calls["n"] == 3
    assert ingestion.DEAD_LETTER_QUEUE and ingestion.DEAD_LETTER_QUEUE[0][0] == ing_event
