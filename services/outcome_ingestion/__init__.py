import hashlib
import os

import planner
from backend.outcomes import OutcomeEvent, save_outcome_event


def _eligible_for_ingestion(account_id: str) -> bool:
    """Return ``True`` if planner should process outcomes for ``account_id``."""

    if os.getenv("ENABLE_OUTCOME_INGESTION", "true").lower() not in {
        "1",
        "true",
        "yes",
    }:
        return False

    percent_raw = os.getenv("OUTCOME_INGESTION_CANARY_PERCENT", "100")
    try:
        percent = float(percent_raw)
    except ValueError:
        percent = 0.0

    if percent >= 100:
        return True
    if percent <= 0:
        return False

    digest = hashlib.sha256(account_id.encode()).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return bucket < percent


def ingest(session: dict, event: OutcomeEvent) -> None:
    """Persist an outcome event and update planner state."""

    session_id = session.get("session_id")
    if not session_id:
        return

    save_outcome_event(session_id, event)

    acc_id = getattr(event, "account_id", "")
    if _eligible_for_ingestion(str(acc_id)):
        planner.handle_outcome(session, event)
