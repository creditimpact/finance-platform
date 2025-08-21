from dataclasses import dataclass

@dataclass
class OutcomeEvent:
    """Record produced when a bureau responds to a dispute."""

    outcome_id: str
    account_id: str
    cycle: int
    result: str
