from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional


class AccountStatus(str, Enum):
    PLANNED = "Planned"
    SENT = "Sent"
    CRA_RESPONDED_VERIFIED = "CRA_RespondedVerified"
    CRA_RESPONDED_UPDATED = "CRA_RespondedUpdated"
    CRA_RESPONDED_DELETED = "CRA_RespondedDeleted"
    CRA_RESPONDED_NOCHANGE = "CRA_RespondedNoChange"
    COMPLETED = "Completed"


@dataclass
class StateTransition:
    """Record of a state change for auditing."""

    from_status: AccountStatus
    to_status: AccountStatus
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AccountState:
    """Tracks the current state of an account in the strategy pipeline."""

    account_id: str
    current_cycle: int
    current_step: int
    status: AccountStatus
    last_sent_at: Optional[datetime] = None
    next_eligible_at: Optional[datetime] = None
    history: List[StateTransition] = field(default_factory=list)
