import planner
from backend.outcomes import OutcomeEvent, save_outcome_event


def ingest(session: dict, event: OutcomeEvent) -> None:
    """Persist an outcome event and update planner state."""

    session_id = session.get("session_id")
    if not session_id:
        return
    save_outcome_event(session_id, event)
    planner.handle_outcome(session, event)
