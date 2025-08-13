import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.api.session_manager import get_session, update_intake, update_session
from backend.core.logic.strategy.strategy_engine import generate_strategy


def test_strategy_engine_uses_structured_summaries():
    session_id = "sess-strategy"
    structured = {
        "1": {
            "account_id": "1",
            "dispute_type": "late",
            "facts_summary": "summary",
            "claimed_errors": [],
            "dates": {},
            "evidence": [],
            "risk_flags": {},
        }
    }
    update_session(session_id, structured_summaries=structured)
    update_intake(session_id, raw_explanations=[{"account_id": "1", "text": "raw"}])
    strategy = generate_strategy(session_id, {"Experian": {"disputes": []}})
    assert strategy["dispute_items"] == structured
    # ensure per-account strategy items were produced
    assert any(item["account_id"] == "1" for item in strategy.get("items", []))
    session = get_session(session_id)
    assert "strategy" in session
    assert "raw_explanations" not in session
    assert "raw" not in json.dumps(strategy)


def test_strategy_engine_assigns_legal_basis_and_tone():
    session_id = "sess-legal"
    structured = {
        "1": {
            "account_id": "1",
            "dispute_type": "late",
            "facts_summary": "I was never late",
        },
        "2": {
            "account_id": "2",
            "dispute_type": "identity_theft",
            "facts_summary": "Not my account",
        },
    }
    update_session(session_id, structured_summaries=structured)

    bureau_data = {
        "Experian": {
            "disputes": [
                {"account_id": "1", "name": "ACME", "status": "Late"},
                {"account_id": "2", "name": "Fraud Co", "status": "Collections"},
            ]
        }
    }

    strategy = generate_strategy(session_id, bureau_data)
    assert len(strategy.get("items", [])) == 2
    item1 = next(i for i in strategy["items"] if i["account_id"] == "1")
    item2 = next(i for i in strategy["items"] if i["account_id"] == "2")
    assert item1["legal_basis"] == "FCRA 611(a)"
    assert item2["legal_basis"] == "FCRA 605B"
    assert item1["tone"] != item2["tone"]
    assert strategy["follow_up"]
