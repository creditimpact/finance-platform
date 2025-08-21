from typing import List

import planner
from backend.outcomes import OutcomeEvent

_events: List[OutcomeEvent] = []


def ingest(session: dict, event: OutcomeEvent) -> None:
    """Persist an outcome event and update planner state."""
    _events.append(event)
    planner.handle_outcome(session, event)


def get_events() -> List[OutcomeEvent]:
    """Return ingested outcome events."""
    return list(_events)
