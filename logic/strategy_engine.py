from __future__ import annotations
from datetime import datetime
from typing import Any, Dict

from session_manager import get_session, update_session
from .rules_loader import load_rules
from .outcomes_store import get_outcomes


def generate_strategy(session_id: str, bureau_data: Dict[str, Any]) -> Dict[str, Any]:
    """Build a strategy document for a given session.

    The strategy combines the sanitized explanations (stored as
    ``structured_summaries``), the current rulebook, a snapshot of the
    provided credit report data and recent outcome telemetry. The raw client
    explanations are intentionally excluded to prevent accidental leakage
    into any generated letters.
    """

    session = get_session(session_id) or {}
    structured = session.get("structured_summaries", {})

    strategy: Dict[str, Any] = {
        "generated_at": datetime.utcnow().isoformat(),
        "rules": load_rules(),
        "dispute_items": structured,
        "bureau_data": bureau_data,
        "historical_outcomes": get_outcomes(),
    }

    update_session(session_id, strategy=strategy)
    return strategy
