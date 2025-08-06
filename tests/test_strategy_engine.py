import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from session_manager import update_session, get_session
from logic.strategy_engine import generate_strategy


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
    update_session(session_id, structured_summaries=structured, raw_explanations=[{"account_id": "1", "text": "raw"}])
    strategy = generate_strategy(session_id, {"Experian": {"disputes": []}})
    assert strategy["dispute_items"] == structured
    session = get_session(session_id)
    assert "strategy" in session
    assert "raw" not in json.dumps(strategy)
